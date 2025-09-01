from __future__ import print_function

import itertools
import os

import torch
from torchvision.models import vgg19

from Feature_Inversion_Generators import *
from new_model import ULikeNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.optim as optim
from torchsummary import summary
from torchvision import transforms
import shutil
from mid_model_packages import *
from dpsgd_model_packages import *
import sys
from InfoDR_package import *
from tqdm import tqdm
from target_model_packages import *
from inversion_model_packages import *
from utilis import *

import torchvision.utils as vutils
from datetime import datetime
import re
from Entropy_Enhancer_Net import *

current_time = datetime.now()

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch_size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='')
parser.add_argument('--epochs', type=int, default=100, metavar='')
parser.add_argument('--lr', type=float, default=0.01, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--allocation', type=float, default=0.8, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
parser.add_argument('--scale_factor', type=int, default=8)
parser.add_argument('--block_idx', type=int, default=1)
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--nz', type=int, default=2)
parser.add_argument('--truncation', type=int, default=2)
parser.add_argument('--sigma2', type=int, default=24)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument('--dataset', default='mnist', metavar='')
parser.add_argument('--user', default='lenovo', metavar='')
parser.add_argument('--target_model', default='cnn', help='cnn | vgg | ir152 | resnet')
parser.add_argument('--method', default='mas', help='inversion | amplify')
parser.add_argument('--optimize', default='adam', help='sgd | adam')
parser.add_argument('--img_size', type=int, default=64)
parser.add_argument('--set_type', default='sub', help='sub | cross')
parser.add_argument('--scheduler', default='yes', help='yes | no')

torch.autograd.set_detect_anomaly(True)


class Logger(object):
    def __init__(self, filename="mnist_classifier.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_pth_paths(base_dir, dataset, target_model):
    full_path = None
    # 定义匹配目标模型的正则表达式
    pattern = re.compile(rf'{re.escape(dataset)}_(?P<target_model>.+?)\.pth')

    # 遍历主文件夹
    for root, _, files in os.walk(base_dir):
        for file in files:
            # 只匹配 .pth 文件
            if file.endswith(".pth"):
                match = pattern.match(file)
                if match:
                    file_target_model = match.group('target_model')
                    # 匹配条件
                    if file_target_model.startswith(target_model):
                        # 构建完整路径
                        full_path = os.path.join(root, file)
                        return full_path


def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    psnr_value = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr_value.item()


def cosine_similarity(tensor1, tensor2):
    cos_sim = F.cosine_similarity(tensor1.view(tensor1.size(0), -1), tensor2.view(tensor2.size(0), -1), dim=1)
    return cos_sim.mean().item()


def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow(x[:,:,1:,:]-x[:,:,:h_x-1,:], 2).sum()
    w_tv = torch.pow(x[:,:,:,1:]-x[:,:,:,:w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def compute_feature_stats(dataloader, model, device):
    args = parser.parse_args()
    model.eval()
    all_features = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            features, _ = model(data)  # 提取 z 特征
            z = features[args.block_idx-1]
            all_features.append(z.cpu())

    features_cat = torch.cat(all_features, dim=0)  # [N, C, H, W]
    mean = features_cat.mean(dim=(0, 2, 3))  # 每个通道均值
    std = features_cat.std(dim=(0, 2, 3)) + 1e-8  # 每个通道标准差
    return mean.cuda(), std.cuda()


class FeatureNormalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std", std.view(1, -1, 1, 1))

    def forward(self, z):
        z = (z - self.mean) / self.std
        return z.cuda()


def train(classifier, inversion, log_interval, device, data_loader, optimizer, epoch):
    args = parser.parse_args()
    classifier.eval()
    inversion.train()

    if args.nc == 1:
        transform_amplify = transforms.Compose([transforms.Normalize((0.5,), (0.5,))
                                              ])
    else:
        transform_amplify = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ])

    start_time = time.time()

    total_mse_loss = 0
    total_psnr = 0
    count = 0
    with tqdm(total=len(data_loader), desc=f"Epoch {epoch}", unit="batch") as pbar:
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                feat, prediction = classifier(transform_amplify(data), block_idx=args.block_idx, release=True)
                # feat, mu, std, prediction = classifier(transform_amplify(data), release=True)

            # print(len(feat))
            # exit()

            z = feat[args.block_idx - 1]
            # z = z[:,1:,:]
            # print(z.shape)
            # exit()
            # z_aug, reconstruction = inversion(z)
            # reconstruction = inversion(z)
            # fn = FeatureNormalizer(mean, std)
            # reconstruction = inversion(fn(z))
            reconstruction = inversion(z)

            psnr_value = psnr(reconstruction, data, max_val=1.0)
            total_psnr += psnr_value

            # z_feat, _ = classifier(transform_amplify(reconstruction), block_idx=args.block_idx, release=True)
            # feat_mse = F.mse_loss(z, z_feat[args.block_idx - 1])

            # z_loss = - distance_correlation(data, z_aug)
            mse_loss = F.mse_loss(reconstruction, data)

            total_mse_loss += mse_loss.item()

            count += 1

            # total_loss = mse_loss + z_loss
            total_loss = mse_loss
            total_loss.backward()
            optimizer.step()

            # 计算已用时间
            elapsed_time = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

            # 更新进度条
            # pbar.set_postfix(mse_loss=mse_loss.item(), z_loss=z_loss.item(), time=elapsed_str)
            pbar.set_postfix(mse_loss=mse_loss.item(), time=elapsed_str, total_mse_loss=total_mse_loss)
            pbar.update()

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}, Learning Rate: {current_lr}')
    avg_psnr = total_psnr / count
    avg_mse = args.nc * total_mse_loss / count
    print('\nTest inversion model on {} set: Average MSE loss: {:.6f}, Average PSNR loss: {:.6f}, '.format(epoch, avg_mse, avg_psnr,))


def test(classifier, inversion, device, data_loader, epoch, msg):
    args = parser.parse_args()
    classifier.eval()
    inversion.eval()

    if args.nc == 1:
        transform_amplify = transforms.Compose([transforms.Normalize((0.5,), (0.5,))
                                                ])
    else:
        transform_amplify = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ])

    mse_loss = 0
    total_psnr = 0
    total_ssim = 0
    total_cosine_similarity = 0
    count = 0
    plot = True
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            feat, prediction = classifier(transform_amplify(data), block_idx=args.block_idx, release=True)
            # feat, mu, std, prediction = classifier(transform_amplify(data), release=True)
            z = feat[args.block_idx - 1]
            # z = z[:, 1:,:]
            # print(z.shape)
            # exit()
            # z_aug, reconstruction = inversion(z)
            # reconstruction = inversion(z)
            # fn = FeatureNormalizer(mean, std)
            # fn = FeatureNormalizer(mean, std)
            # reconstruction = inversion(fn(z))
            reconstruction = inversion(z)

            mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

            psnr_value = psnr(reconstruction, data, max_val=1.0)
            total_psnr += psnr_value

            # if args.nc == 1:
            #     ssim_value = ssim(reconstruction.cpu().numpy(), data.cpu().numpy(), data_range=1.0, channel_axis=1, multichannel=False)
            # else:
            #     ssim_value = ssim(reconstruction.cpu().numpy(), data.cpu().numpy(), data_range=1.0, channel_axis=1, multichannel=True)
            # total_ssim += ssim_value

            cosine_sim = cosine_similarity(reconstruction, data)
            total_cosine_similarity += cosine_sim

            count += 1

            if plot:
                truth = data[0:32]
                inverse = reconstruction[0:32]
                out = torch.cat((truth, inverse))

                for i in range(4):
                    out[i * 16:i * 16 + 8] = truth[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = inverse[i * 8:i * 8 + 8]

                out = out.flip(1)
                # Save image
                os.makedirs(f'out/{args.dataset}/{args.target_model}/{args.block_idx}', exist_ok=True)
                vutils.save_image(out, 'out/{}/{}/{}/recon_{}_{}.png'.format(args.dataset, args.target_model,
                                                                               args.block_idx, msg.replace(" ", ""), epoch),
                                  normalize=False)
                plot = False

    mse_loss /= len(data_loader.dataset) * args.img_size * args.img_size
    avg_psnr = total_psnr / count
    # avg_ssim = total_ssim / count
    avg_cosine_similarity = total_cosine_similarity / count
    print('\nTest inversion model on {} set: Average MSE loss: {:.6f}, Average PSNR loss: {:.6f}, '
          'Average Cosine Similarity: {:.6f}\n'.format(msg, mse_loss, avg_psnr,
                                                                                  avg_cosine_similarity))
    # print('\nTest inversion model on {} set: Average MSE loss: {:.6f}, Average PSNR loss: {:.6f}, '
    #       'Average SSIM loss: {:.6f},Average Cosine Similarity: {:.6f}\n'.format(msg, mse_loss, avg_psnr,
    #                                                                              avg_ssim, avg_cosine_similarity))
    return mse_loss


def evaluate_inversion_model(inversion, classifier, test_loader, device):
    args = parser.parse_args()
    classifier.eval()
    inversion.eval()

    if args.nc == 1:
        transform_amplify = transforms.Compose([transforms.Normalize((0.5,), (0.5,))
                                                ])
    else:
        transform_amplify = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ])
    total_mse, total_psnr, total_ssim, total_cos = 0, 0, 0, 0
    num_samples = 0

    with torch.no_grad():
        for data in test_loader:
            images, _ = data  # 假设 test_loader 提供的是图像和标签
            images = images.to(device)

            # 使用 Inversion 模型生成重建图像
            feat, prediction = classifier(transform_amplify(images), block_idx=args.block_idx, release=True)
            # feat, mu, std, prediction = classifier(transform_amplify(images), release=True)
            z = feat[args.block_idx - 1]
            # z = z[:,1:,:]
            # print(z.shape)
            # exit()
            # z_aug, reconstruction = inversion(z)
            #　reconstruction = inversion(z)
            # fn = FeatureNormalizer(mean, std)
            # fn = FeatureNormalizer(mean, std)
            # reconstruction = inversion(fn(z))
            reconstruction = inversion(z)

            # 计算 MSE
            total_mse += F.mse_loss(reconstruction, images, reduction='sum').item()

            # 计算 PSNR
            psnr_value = psnr(reconstruction, images, max_val=1.0)
            total_psnr += psnr_value

            # 计算 SSIM
            if args.nc == 1:
                ssim_value = ssim(reconstruction.cpu().numpy(), images.cpu().numpy(), data_range=1.0, channel_axis=1, multichannel=False)
            else:
                ssim_value = ssim(reconstruction.cpu().numpy(), images.cpu().numpy(), data_range=1.0, channel_axis=1, multichannel=True)
            total_ssim += ssim_value

            # 计算 Cosine Similarity
            cosine_sim = cosine_similarity(reconstruction, images)
            total_cos += cosine_sim

            num_samples += 1

    # 计算平均值
    total_mse /= len(test_loader.dataset) * args.img_size * args.img_size
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_cos = total_cos / num_samples

    return total_mse, avg_psnr, avg_ssim, avg_cos


def main():
    args = parser.parse_args()
    os.makedirs(f'log/attack/{args.dataset}', exist_ok=True)
    sys.stdout = Logger(f'./log/attack/{args.dataset}/{args.dataset}_{args.target_model}_{args.block_idx}_{args.lr}_train_log.txt')
    print("Formatted time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("================================")
    print(args)
    print("================================")
    os.makedirs('out', exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)

    transform = transforms.Compose([transforms.ToTensor(),])

    if args.dataset == 'facescrub':
        train_set = CelebA(root=f'/home/{args.user}/', transform=transform)
        # Inversion attack on TRAIN data of facescrub classifier
        test1_set = FaceScrub(root=f'/home/{args.user}/', transform=transform, train=True)
        # Inversion attack on TEST data of facescrub classifier
        test2_set = FaceScrub(root=f'/home/{args.user}/', transform=transform, train=False)

    elif args.dataset == 'chest':
        train_set = Chest(root=f'/home/{args.user}/', transform=transform, attack=True)
        # Inversion attack on TRAIN data of chest classifier
        test1_set = Chest(root=f'/home/{args.user}/', transform=transform, train=True)
        # Inversion attack on TEST data of chest classifier
        test2_set = Chest(root=f'/home/{args.user}/', transform=transform, train=False)

    elif args.dataset == 'cifar10':
        train_set = CIFAR10_64(root=f'/home/{args.user}/', attack=True,
                               transform=transform, )
        # Inversion attack on TRAIN data of cifar64 classifier
        test1_set = CIFAR10_64(root=f'/home/{args.user}/', train=True, transform=transform)
        # Inversion attack on TEST data of cifar64 classifier
        test2_set = CIFAR10_64(root=f'/home/{args.user}/', train=False, transform=transform)

    elif args.dataset == 'mnist':
        train_set = KMNIST(root=f'/home/{args.user}/', transform=transform, attack=True)
        # Inversion attack on TRAIN data of chest classifier
        test1_set = MNIST(root=f'/home/{args.user}/', transform=transform, train=True)
        # Inversion attack on TEST data of chest classifier
        test2_set = MNIST(root=f'/home/{args.user}/', transform=transform, train=False)

    elif args.dataset == 'emnist':
        train_set = EMNIST(root=f'/home/{args.user}/', transform=transform, attack=True)
        # Inversion attack on TRAIN data of chest classifier
        test1_set = EMNIST(root=f'/home/{args.user}/', transform=transform, train=True)
        # Inversion attack on TEST data of chest classifier
        test2_set = EMNIST(root=f'/home/{args.user}/', transform=transform, train=False)

    elif args.dataset == 'kmnist':
        train_set = MNIST(root=f'/home/{args.user}/', transform=transform, attack=True)
        # Inversion attack on TRAIN data of chest classifier
        test1_set = KMNIST(root=f'/home/{args.user}/', transform=transform, train=True)
        # Inversion attack on TEST data of chest classifier
        test2_set = KMNIST(root=f'/home/{args.user}/', transform=transform, train=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    # test1_loader = torch.utils.data.DataLoader(test1_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    test2_loader = torch.utils.data.DataLoader(test2_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # The definition of target model
    if args.nc == 1:
        transform_amplify = transforms.Compose([transforms.Normalize((0.5,), (0.5,))
                                              ])
    else:
        transform_amplify = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ])
    if args.target_model == "cnn":
        classifier = Classifier_4(nc=args.nc, ndf=args.ndf, nz=args.nz, img_size=args.img_size).to(device)
        # classifier_cpu = Classifier_4(nc=args.nc, ndf=args.ndf, nz=args.nz, img_size=args.img_size).to("cpu")
        # Define inversion model
        output_size_dict = {
            1: (32, 32),
            2: (16, 16),
            3: (16, 16),
            4: (4, 4)
        }
        input_channels_dict = {
            1: args.ngf,  # Assuming the output channels of block 1 are ngf
            2: args.ngf * 2,  # Assuming the output channels of block 2 are ngf * 2
            3: 256,  # Assuming the output channels of block 3 are ngf * 4
            4: args.ngf * 8  # Assuming the output channels of block 4 are ngf * 8
        }
        # output_size_dict = {
        #     1: (32, 32),
        #     2: (16, 16),
        #     3: (16, 16),
        #     4: (8, 8)
        # }
        # input_channels_dict = {
        #     1: args.ngf,  # Assuming the output channels of block 1 are ngf
        #     2: args.ngf,  # Assuming the output channels of block 2 are ngf * 2
        #     3: args.ngf * 2,  # Assuming the output channels of block 3 are ngf * 4
        #     4: args.ngf * 4  # Assuming the output channels of block 4 are ngf * 8
        # }
        # Get the output size and input channels for the current block_idx
        output_size = output_size_dict.get(args.block_idx, (32, 32))  # Default to (32, 32) if block_idx is not found
        input_channels = input_channels_dict.get(args.block_idx,
                                                 args.ngf)  # Default to args.ngf if block_idx is not found
        input_size = (input_channels, *output_size)
        inversion = Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.nz, c=50).to(device)
        # input_size = (input_channels, *output_size)
        # input_size = (530,)
        # input_size = (1, args.nz)
        # Print model summary
        summary(inversion, input_size)

        # input_data = torch.randn(128, args.nc, args.img_size, args.img_size).to(device)
        # input_data_cpu = torch.randn(128, args.nc, args.img_size, args.img_size).to("cpu")
        # _, times, memories = classifier(transform_amplify(input_data), block_idx=args.block_idx, record_time_memory=True)
        # print("GPU Time for block {}: {} ms".format(args.block_idx, sum(times[:args.block_idx])))
        # print("GPU Video memory for block {}: {} MB".format(args.block_idx, sum(memories[:args.block_idx])))

        # _, times, memories = classifier_cpu(transform_amplify(input_data_cpu), block_idx=args.block_idx,
        #                                 record_time_memory=True, device="cpu")
        # print("CPU Time for block {}: {} ms".format(args.block_idx, sum(times[:args.block_idx])))
        # print("CPU Memory for block {}: {} MB".format(args.block_idx, sum(memories[:args.block_idx])))

        optimizer = optim.Adam(inversion.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                               verbose=True)

    elif args.target_model == "resnet":
        classifier = rn18(num_classes=args.nz, nc=args.nc, img_size=args.img_size).to(device)
        # resnet_profiler = rn18_test(num_classes=args.nz, nc=args.nc, img_size=args.img_size).to(device)
        # resnet_profiler_cpu = rn18_test(num_classes=args.nz, nc=args.nc, img_size=args.img_size).to("cpu")
        # resnet_profiler.profile(input_size=(64, args.nc, args.img_size, args.img_size)
        #                         , block_indices=args.block_idx, device="cuda")
        # resnet_profiler_cpu.profile(input_size=(64, args.nc, args.img_size, args.img_size)
        #                         , block_indices=args.block_idx, device="cpu")

        inversion = inversion_resnet_pv4(block_idx=args.block_idx, num_classes=args.nz, nc=args.nc,
                                     img_size=args.img_size).to(device)
        # Define inversion model
        output_size_dict = {
            1: (64, 64),
            2: (64, 64),
            3: (64, 64),
            4: (8, 8)
        }
        input_channels_dict = {
            1: args.ngf,  # Assuming the output channels of block 1 are ngf
            2: 2,  # Assuming the output channels of block 2 are ngf * 2
            3: 64,  # Assuming the output channels of block 3 are ngf * 4
            4: args.ngf * 8  # Assuming the output channels of block 4 are ngf * 8
        }
        # Get the output size and input channels for the current block_idx
        if args.block_idx <= 3:
            output_size = output_size_dict.get(2, (64, 64))  # Default to (32, 32) if block_idx is not found
            input_channels = input_channels_dict.get(2, 64)  # Default to args.ngf if block_idx is not found
            input_size = (input_channels, *output_size)
            # Print model summary
            summary(inversion, input_size)
        if args.block_idx in range(4, 6):
            output_size = output_size_dict.get(2, (32, 32))  # Default to (32, 32) if block_idx is not found
            input_channels = input_channels_dict.get(2, args.ngf)  # Default to args.ngf if block_idx is not found
            input_size = (input_channels, *output_size)
            # Print model summary
            summary(inversion, input_size)

        if args.block_idx in range(6, 8):
            output_size = output_size_dict.get(3, (32, 32))  # Default to (32, 32) if block_idx is not found
            input_channels = input_channels_dict.get(3, args.ngf)  # Default to args.ngf if block_idx is not found
            input_size = (input_channels, *output_size)
            # Print model summary
            summary(inversion, input_size)

        if args.block_idx in range(8, 11):
            output_size = output_size_dict.get(4, (32, 32))  # Default to (32, 32) if block_idx is not found
            input_channels = input_channels_dict.get(4, args.ngf)  # Default to args.ngf if block_idx is not found
            input_size = (input_channels, *output_size)
            # Print model summary
            summary(inversion, input_size)

        optimizer = optim.Adam(inversion.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                               verbose=True)


    elif args.target_model == 'vgg':
        classifier = vgg19(nc=args.nc, nz=args.nz, img_size=args.img_size).to(device)
        # Define inversion model
        output_size_dict = {
            1: (32, 32),
            2: (32, 32),
            3: (16, 16),
            4: (4, 4),
            13: (32, 32),
            17: (16, 16),
            18: (16, 16),
            26: (16, 16),
            30: (8, 8),
            39: (8, 8),
            43: (4, 4),
            52: (4, 4),
        }
        input_channels_dict = {
            1: args.ngf,  # Assuming the output channels of block 1 are ngf
            2: 4,  # Assuming the output channels of block 2 are ngf * 2
            3: 12,  # Assuming the output channels of block 3 are ngf * 4
            4: args.ngf * 8,  # Assuming the output channels of block 4 are ngf * 8
            13: 128,
            17: 256,
            18: 128,
            26: 256,
            39: 512,
            30: 512,
            43: 512,
            52: 512
        }

        output_size = output_size_dict.get(args.block_idx, (16, 16))  # Default to (32, 32) if block_idx is not found
        input_channels = input_channels_dict.get(args.block_idx,
                                                 12)  # Default to args.ngf if block_idx is not found
        input_size = (input_channels, *output_size)
        inversion = DecoderForIR152(block_idx=args.block_idx, in_channels=input_channels, image_channels=args.nc).to(device)
        # inversion = DecoderWithAugmentation(in_channels=input_channels, image_channels=args.nc).to(device)
        # input_size = (1, args.nz)
        # Print model summary
        summary(inversion, input_size)

        # input_data = torch.randn(128, args.nc, args.img_size, args.img_size).to(device)
        # input_data_cpu = torch.randn(128, args.nc, args.img_size, args.img_size).to("cpu")
        # _, times, memories = classifier(transform_amplify(input_data), block_idx=args.block_idx, record_time_memory=True)
        # print("GPU Time for block {}: {} ms".format(args.block_idx, sum(times[:args.block_idx])))
        # print("GPU Video memory for block {}: {} MB".format(args.block_idx, sum(memories[:args.block_idx])))

        # _, times, memories = classifier_cpu(transform_amplify(input_data_cpu), block_idx=args.block_idx,
        #                                 record_time_memory=True, device="cpu")
        # print("CPU Time for block {}: {} ms".format(args.block_idx, sum(times[:args.block_idx])))
        # print("CPU Memory for block {}: {} MB".format(args.block_idx, sum(memories[:args.block_idx])))

        optimizer = optim.Adam(inversion.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8,
                                                               verbose=True)

    elif args.target_model == "ir152":
        classifier = IR152(args.nz, args.nc, args.img_size).to(device)
        # Define inversion model
        output_size_dict = {
            3: (32, 32),
            6: (16, 16),
            8: (16, 16),
            12: (16, 16),
            11: (16, 16),
            # 12: (8, 8),
            25: (8, 8),
            35: (8, 8),
            40: (8, 8),
            47: (8, 8),
            48: (8, 8),
            49: (4, 4),
            50: (4, 4),
            51: (4, 4),
        }
        input_channels_dict = {
            3: 64,  # Assuming the output channels of block 1 are ngf
            6: 128,  # Assuming the output channels of block 2 are ngf * 2
            8: 128,  # Assuming the output channels of block 3 are ngf * 4
            12: 2,
            11: 128,  # Assuming the output channels of block 4 are ngf * 8
            # 12: 256,
            25: 256,
            35: 256,
            40: 256,
            47: 256,
            48: 256,
            49: 512,
            50: 512,
            51: 512,
        }

        output_size = output_size_dict.get(args.block_idx, (16, 16))  # Default to (32, 32) if block_idx is not found
        input_channels = input_channels_dict.get(args.block_idx,
                                                 128)  # Default to args.ngf if block_idx is not found
        # inversion = ULikeNet(in_channels=input_channels, out_channels=args.nc).to(device)
        # inversion = inversion_ir152(nc=args.nc, ngf=args.ngf, ndf=args.ndf, nz=args.nz, img_size=64, block_idx=args.block_idx).to(device)
        # inversion = StyleTinyInverter(style_dim=input_channels, fmap_base=args.img_size).to(device)
        # inversion = SPADEUNetInverter(base_ch=args.img_size).to(device)
        # inversion = CascadeRefineNet(base_ch=128).to(device)
        inversion = DecoderWithAugmentation(in_channels=input_channels, image_channels=args.nc).to(device)
        #inversion = DecoderForIR152AE(block_idx=args.block_idx, in_channels=input_channels, image_channels=args.nc).to(
        #     device)
        # inversion = DecoderForIR152(block_idx=args.block_idx, in_channels=input_channels, image_channels=args.nc).to(device)
        # inversion = Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.nz, c=50).to(device)
        input_size = (input_channels, *output_size)
        # input_size = (10, )
        # input_size = (1, args.nz)
        # Print model summary
        summary(inversion, input_size)

        # optimizer = optim.Adam(inversion.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
        optimizer = optim.AdamW(inversion.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8,
                                                               verbose=True)

    elif args.target_model == "vit":
        classifier = VisionTransformer(img_size=64, patch_size=4, embed_dim=128, depth=6, num_heads=8, num_classes=10).to(device)
        # Define inversion model

        output_size = 128  # Default to (32, 32) if block_idx is not found
        input_channels = 256
        # inversion = ULikeNet(in_channels=input_channels, out_channels=args.nc).to(device)
        # inversion = inversion_ir152(nc=args.nc, ngf=args.ngf, ndf=args.ndf, nz=args.nz, img_size=64, block_idx=args.block_idx).to(device)
        # inversion = StyleTinyInverter(style_dim=input_channels, fmap_base=args.img_size).to(device)
        # inversion = SPADEUNetInverter(base_ch=args.img_size).to(device)
        inversion = InversionForVIT(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.nz, c=50).to(device)
        # inversion = DecoderWithAugmentation(in_channels=input_channels, image_channels=args.nc).to(device)
        #inversion = DecoderForIR152AE(block_idx=args.block_idx, in_channels=input_channels, image_channels=args.nc).to(
        #     device)
        # inversion = DecoderForIR152(block_idx=args.block_idx, in_channels=input_channels, image_channels=args.nc).to(device)
        # inversion = Inversion(nc=args.nc, ngf=args.ngf, nz=args.nz, truncation=args.nz, c=50).to(device)
        input_size = (input_channels, output_size)
        # input_size = (10, )
        # input_size = (1, args.nz)
        # Print model summary
        summary(inversion, input_size)

        optimizer = optim.Adam(inversion.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
        # optimizer = optim.AdamW(inversion.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8,
                                                               verbose=True)


    # Load classifier
    os.makedirs(f'inversion_model/{args.dataset}/{args.target_model}/finding6', exist_ok=True)

    base_dir = f'/home/{args.user}/'
    # base_dir = f'/home/{args.user}/桌面/liurk/CORECODE/CODE/info_protect_target_model'
    # path = get_pth_paths(base_dir, args.dataset, args.target_model)
    path = "/home/yons/"
    print(path)
    checkpoint = torch.load(path, weights_only=True)

    state_dict = checkpoint['model']
    state_dict = {k.replace('_module.', ''): v for k, v in state_dict.items()}
    classifier.load_state_dict(state_dict)

    # classifier.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    best_cl_acc = checkpoint['best_cl_acc']
    print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))


    # Train inversion model
    best_recon_loss = 999
    recon_loss_train_list = []
    recon_loss_test_list = []
    best_epoch = 0
    # mean, std = compute_feature_stats(train_loader, classifier, device)
    for epoch in range(1, args.epochs + 1):
        train(classifier, inversion, args.log_interval, device, train_loader, optimizer, epoch)
        # recon_loss_train = test(classifier, inversion, device, test1_loader, epoch, 'test1')
        recon_loss_test = test(classifier, inversion, device, test2_loader, epoch, 'test2')
        if args.scheduler == "yes":
            scheduler.step(recon_loss_test)

        recon_loss_train_list.append(recon_loss_test)
        recon_loss_test_list.append(recon_loss_test)

        if recon_loss_test < best_recon_loss:
            best_epoch = epoch
            best_recon_loss = recon_loss_test
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_recon_loss': best_recon_loss
            }
            torch.save(state, f'inversion_model/{args.dataset}/{args.target_model}/finding6/inversion_{args.block_idx}_kmnist.pth')
            # shutil.copyfile('out/{}/{}/{}/recon_test1_{}.png'.format(args.dataset, args.target_model, args.block_idx, epoch),
            #                 f'out/{args.dataset}/{args.target_model}/{args.block_idx}/best_test1.png')
            shutil.copyfile('out/{}/{}/{}/recon_test2_{}.png'.format(args.dataset, args.target_model,  args.block_idx, epoch),
                            f'out/{args.dataset}/{args.target_model}/{args.block_idx}/best_test2.png')

        print("For now, the best Epoch is {}".format(best_epoch))

    print(f'Epoch {epoch} gets the best reconstruction loss: {best_recon_loss:.6f}')

    # 在训练完成后进行评估
    best_model_path = f'inversion_model/{args.dataset}/{args.target_model}/finding6/inversion_{args.block_idx}_kmnist.pth'
    checkpoint = torch.load(best_model_path, weights_only=True)
    inversion.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    best_recon_loss = checkpoint['best_recon_loss']
    print("=> loaded inversion checkpoint '{}' (epoch {}, acc {:.4f})".format(best_model_path, epoch, best_recon_loss))

    # 使用 test2_loader 进行评估
    avg_mse, avg_psnr, avg_ssim, avg_cos = evaluate_inversion_model(inversion, classifier, test2_loader, device)

    # 打印结果
    print(
        f"Best Inversion Network Evaluation:\nMSE: {avg_mse:.6f}, PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}, Cosine Similarity: {avg_cos:.6f}")

    # Plotting recon loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), recon_loss_train_list, label='Train set Loss')
    plt.plot(range(1, args.epochs + 1), recon_loss_test_list, label='Test set Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Reconstruction Loss Over Epochs')
    plt.legend()
    plt.savefig(f'inversion_model/{args.dataset}/{args.target_model}/inversion_{args.block_idx}.jpg')
    plt.close()
    print("Formatted time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == '__main__':
    main()
