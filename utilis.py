from __future__ import print_function, division

import math
import os
import cv2
import torch.nn as nn
import torch
import torchvision
import numpy as np
from skimage import exposure
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

np.random.seed(624)


class CIFAR10_64(Dataset):
    def __init__(self, root, random_numbers=None, transform=None, target_transform=None, allocation=0.73, data_type=None,
                 train=True, test_free=True, attack=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(os.path.join(self.root, 'cifar10_data.npz'))
        train_data = input['train_data']
        train_labels = input['train_labels']
        test_data = input['test_data']
        test_labels = input['test_labels']
        auxiliary_data = input['auxiliary_data']
        auxiliary_labels = input['auxiliary_labels']

        print(f"Length of training data: {len(train_data)}")
        print(f"Length of auxiliary data: {len(auxiliary_data)}")
        print(f"Length of test data: {len(test_data)}")

        if train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels

        if attack:
            self.data = auxiliary_data
            self.labels = auxiliary_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FaceScrub(Dataset):
    def __init__(self, root, random_numbers=None, transform=None, target_transform=None, allocation=0.8, data_type=None,
                 train=True, test_free=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(os.path.join(self.root, 'facescrub_color1.npz'))
        actor_images = input['actor_images']
        actor_labels = input['actor_labels']
        if data_type == 'f_m':
            actor_labels.fill(0)
        elif data_type == 'bi':
            actor_labels = np.where(np.isin(actor_labels, random_numbers), 1, 0)
        actress_images = input['actress_images']
        actress_labels = input['actress_labels']
        if data_type == 'f_m':
            actress_labels.fill(1)
        elif data_type == 'bi':
            actress_labels = np.where(np.isin(actress_labels, random_numbers), 1, 0)

        data = np.concatenate([actor_images, actress_images], axis=0)
        labels = np.concatenate([actor_labels, actress_labels], axis=0).astype(np.int64)

        # v_min = data.min(axis=0)
        # v_max = data.max(axis=0)
        # data = (data - v_min) / (v_max - v_min)

        if test_free:
            np.random.seed(624)
            perm = np.arange(len(data))
            np.random.shuffle(perm)
            data = data[perm]
            labels = labels[perm]

        if train:
            self.data = data[0:int(allocation * len(data))]
            self.labels = labels[0:int(allocation * len(data))]
        else:
            self.data = data[int(allocation * len(data)):]
            self.labels = labels[int(allocation * len(data)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # # 打印类型
        # print(f"Type of img: {type(img)}, Type of target: {type(target)}")
        # print(f"Shape of img: {img.size if isinstance(img, Image.Image) else img.shape}, Value of target: {target}")

        return img, target


class CelebA(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(os.path.join(self.root, 'celebA_crop_color.npz'))
        data = input['celeba']
        # for im in paths:
        #     data.append(np.load(im))
        # data = np.concatenate(data, axis=0)

        # v_min = data.min(axis=0)
        # v_max = data.max(axis=0)
        # data = (data - v_min) / (v_max - v_min)
        self.labels = np.array([0] * len(data))

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class Chest(Dataset):
    def __init__(self, root, random_numbers=None, transform=None, target_transform=None, allocation=0.73, data_type=None,
                 train=True, test_free=True, attack=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(os.path.join(self.root, 'xray_dataset.npz'))
        train_data = input['train_data']
        train_labels = input['train_labels']
        test_data = input['test_data']
        test_labels = input['test_labels']
        auxiliary_data = input['auxiliary_data']
        auxiliary_labels = input['auxiliary_labels']

        if train:
            print(f"Length of training data: {len(train_data)}")
            print(f"Length of auxiliary data: {len(auxiliary_data)}")
        else:
            print(f"Length of test data: {len(test_data)}")

        if train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels

        if attack:
            self.data = auxiliary_data
            self.labels = auxiliary_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MNIST(Dataset):
    def __init__(self, root, random_numbers=None, transform=None, target_transform=None, allocation=0.73, data_type=None,
                 train=True, test_free=True, attack=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(os.path.join(self.root, 'mnist.npz'))
        train_data = input['train_data']
        train_labels = input['train_labels']
        test_data = input['test_data']
        test_labels = input['test_labels']
        auxiliary_data = input['auxiliary_data']
        auxiliary_labels = input['auxiliary_labels']

        if train:
            print(f"Length of training data: {len(train_data)}")
            print(f"Length of auxiliary data: {len(auxiliary_data)}")
        else:
            print(f"Length of test data: {len(test_data)}")

        if train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels

        if attack:
            self.data = auxiliary_data
            self.labels = auxiliary_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class EMNIST(Dataset):
    def __init__(self, root, random_numbers=None, transform=None, target_transform=None, allocation=0.73, data_type=None,
                 train=True, test_free=True, attack=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(os.path.join(self.root, 'emnist.npz'))
        train_data = input['train_data']
        train_labels = input['train_labels']
        test_data = input['test_data']
        test_labels = input['test_labels']
        auxiliary_data = input['auxiliary_data']
        auxiliary_labels = input['auxiliary_labels']

        if train:
            print(f"Length of training data: {len(train_data)}")
            print(f"Length of auxiliary data: {len(auxiliary_data)}")
        else:
            print(f"Length of test data: {len(test_data)}")

        if train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels

        if attack:
            self.data = auxiliary_data
            self.labels = auxiliary_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class KMNIST(Dataset):
    def __init__(self, root, random_numbers=None, transform=None, target_transform=None, allocation=0.73, data_type=None,
                 train=True, test_free=True, attack=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(os.path.join(self.root, 'kmnist.npz'))
        train_data = input['train_data']
        train_labels = input['train_labels']
        test_data = input['test_data']
        test_labels = input['test_labels']
        auxiliary_data = input['auxiliary_data']
        auxiliary_labels = input['auxiliary_labels']

        if train:
            print(f"Length of training data: {len(train_data)}")
            print(f"Length of auxiliary data: {len(auxiliary_data)}")
        else:
            print(f"Length of test data: {len(test_data)}")

        if train:
            self.data = train_data
            self.labels = train_labels
        else:
            self.data = test_data
            self.labels = test_labels

        if attack:
            self.data = auxiliary_data
            self.labels = auxiliary_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def plot_images(images):
    print("testtesttest")
    plt.figure(figsize=(64, 64))
    print("test2")
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    print("test3")
    # plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    ndarr = cv2.medianBlur(ndarr, 1)
    im = Image.fromarray(ndarr)
    im.save(path)


def max_index(lst_int, max_n, release_n=True):
    if release_n:
        # 输出前n大
        index = list(np.argpartition(lst_int, -max_n)[-max_n:])
    else:
        # 输出最大索引
        index = []
        max_n = max(lst_int)
        for i in range(len(lst_int)):
            if lst_int[i] == max_n:
                index.append(i)

    return index  # 返回一个列表


def unmake_grid(grid, nrow=8, padding=2):
    # grid: a 3D Tensor of shape (C x H' x W')
    # nrow: number of images displayed in each row of the grid
    # padding: amount of padding between images
    # return: a 4D Tensor of shape (B x C x H x W) or a list of images

    # get the number of channels, height and width of each image
    c = grid.shape[0]
    h = (grid.shape[1] - (nrow + 1) * padding) // nrow
    w = (grid.shape[2] - (nrow + 1) * padding) // nrow

    # split the grid tensor along the height dimension into nrow chunks
    rows = torch.split(grid, h + padding, dim=1)

    # remove the padding from each row and concatenate them along the channel dimension
    rows = [row[:, padding:-padding, :] for row in rows if row.shape[1] > padding]

    # split the concatenated tensor along the width dimension into B chunks
    images = [torch.split(row, w + padding, dim=2) for row in rows]

    # remove the padding from each image and stack them along the batch dimension
    images = [image[:, :, padding:-padding] for row in images for image in row if image.shape[2] > padding]

    try:
        images = torch.stack(images)
        return images
    except RuntimeError:
        return images


def save_fimages(images, path, **kwargs):
    print("test4")
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).type(torch.uint8).to('cpu').numpy()
    ndarr = exposure.adjust_gamma(ndarr, 2.5)
    print("test5")
    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
    ndarr = cv2.medianBlur(ndarr, 1)
    im = Image.fromarray(ndarr)
    print("test6")
    im.save(path)


def poincare_loss(outputs, targets, xi=1e-4):
    # Normalize logits
    u = outputs / torch.norm(outputs, p=1, dim=-1).unsqueeze(1)
    # Create one-hot encoded target vector
    v = torch.clip(torch.eye(outputs.shape[-1])[targets] - xi, 0, 1).unsqueeze(0)
    v = v.to(u.device)
    # Compute squared norms
    u_norm_squared = torch.norm(u, p=2, dim=1) ** 2
    v_norm_squared = torch.norm(v, p=2, dim=1) ** 2
    diff_norm_squared = torch.norm(u - v, p=2, dim=1) ** 2
    # Compute delta
    delta = 2 * diff_norm_squared / ((1 - u_norm_squared) *
                                     (1 - v_norm_squared))
    # Compute distance
    loss = torch.arccosh(1 + delta)
    return loss


def get_data(args):
    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
    #     torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(64), ])
    dataset = CelebA(args.dataset_path, transform=transform)
    # Inversion attack on TRAIN data of facescrub classifier
    test1_set = FaceScrub(args.dataset_path, transform=transform, train=True)
    # Inversion attack on TEST data of facescrub classifier
    test2_set = FaceScrub(args.dataset_path, transform=transform, train=False)
    # dataset = FaceScrub(args.dataset_path, transform=transforms, train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    test1_loader = DataLoader(test1_set, batch_size=args.test_batch_size, shuffle=False)
    test2_loader = DataLoader(test2_set, batch_size=args.test_batch_size, shuffle=False)
    # dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(64), ])
    # dataset = Mnist(args.dataset_path, transform=transform, train=False)
    # # Inversion attack on TRAIN data of facescrub classifier
    # test1_set = Mnist(args.dataset_path, transform=transform, train=True)
    # # Inversion attack on TEST data of facescrub classifier
    # test2_set = Mnist(args.dataset_path, transform=transform, train=False)
    # # dataset = FaceScrub(args.dataset_path, transform=transforms, train=False)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # test1_loader = DataLoader(test1_set, batch_size=args.test_batch_size, shuffle=False)
    # test2_loader = DataLoader(test2_set, batch_size=args.test_batch_size, shuffle=False)
    # # dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader, test1_loader, test2_loader


def get_testdata(args):
    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
    #     torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    test1_set = FaceScrubtest(args.dataset_path, transform=transforms, train=False, test_free=False)

    test1_loader = DataLoader(test1_set, batch_size=1, shuffle=False)

    return test1_loader


class FaceScrubtest(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True, test_free=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        input = np.load(os.path.join(self.root, 'facescrub_color1.npz'))
        actor_images = input['actor_images']
        actor_labels = input['actor_labels']
        actress_images = input['actress_images']
        actress_labels = input['actress_labels']

        data = np.concatenate([actor_images, actress_images], axis=0)
        labels = np.concatenate([actor_labels, actress_labels], axis=0)
        idx = np.where(actor_labels == 4)
        data = np.array([i for i in actor_images[idx[0][0]:idx[0][-1] + 1]])

        # v_min = data.min(axis=0)
        # v_max = data.max(axis=0)
        # data = (data - v_min) / (v_max - v_min)

        if test_free:
            np.random.seed(666)
            perm = np.arange(len(data))
            np.random.shuffle(perm)
            data = data[perm]
            labels = labels[perm]

        if train:
            self.data = data[0:int(0.8 * len(data))]
            self.labels = labels[0:int(0.8 * len(data))]
        else:
            self.data = data[0:]
            self.labels = labels[0:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, use_label_smoothing=True):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.to(torch.device('cuda'))
        if use_label_smoothing:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class AM_Softmax_v1(nn.Module):  # creates the classification layer
    def __init__(self, m=0.35, s=30, d=2048, num_classes=2, use_gpu=True, epsilon=0.1):
        super(AM_Softmax_v1, self).__init__()
        self.m = m
        self.s = s
        self.num_classes = num_classes

        self.weight = torch.nn.Linear(d, num_classes, bias=False)
        if use_gpu:
            self.weight = self.weight.cuda()
        bound = 1 / math.sqrt(d)
        nn.init.uniform_(self.weight.weight, -bound, bound)
        self.CrossEntropy = CrossEntropyLabelSmooth(self.num_classes, use_gpu=use_gpu)

    def forward(self, x, labels):
        """
        x : feature vector : (b x  d) b= batch size d = dimension
        labels : (b,)
        """
        # x = torch.rand(32,2048)
        # label = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,])
        x = nn.functional.normalize(x, p=2, dim=1)  # normalize the features

        with torch.no_grad():
            self.weight.weight.div_(torch.norm(self.weight.weight, dim=1, keepdim=True))

        b = x.size(0)
        n = self.num_classes

        cos_angle = self.weight(x)
        cos_angle = torch.clamp(cos_angle, min=-1, max=1)
        for i in range(b):
            cos_angle[i][labels[i]] = cos_angle[i][labels[i]] - self.m
        weighted_cos_angle = self.s * cos_angle
        log_probs = self.CrossEntropy(weighted_cos_angle, labels)
        return log_probs


class AM_Softmax_v2(nn.Module):  # requires classification layer for normalization
    def __init__(self, m=0.35, s=30, d=2048, num_classes=625, use_gpu=True, epsilon=0.1):
        super(AM_Softmax_v2, self).__init__()
        self.m = m
        self.s = s
        self.num_classes = num_classes
        self.CrossEntropy = CrossEntropyLabelSmooth(self.num_classes, use_gpu=use_gpu)

    def forward(self, features, labels, classifier):
        '''
        x : feature vector : (b x  d) b= batch size d = dimension
        labels : (b,)
        classifier : Fully Connected weights of classification layer (dxC), C is the number of classes: represents the vectors for class
        '''
        # x = torch.rand(32,2048)
        # label = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,])
        features = nn.functional.normalize(features, p=2, dim=1)  # normalize the features
        with torch.no_grad():
            cnt = 0
            for i in classifier:
                if type(classifier[cnt]) != 'torch.nn.modules.dropout.Dropout' or 'torch.nn.modules.Relu':
                    classifier[cnt].weight.div_(torch.norm(classifier[cnt].weight, dim=1, keepdim=True))

        cos_angle = classifier(features)
        cos_angle = torch.clamp(cos_angle, min=-1, max=1)
        b = features.size(0)
        for i in range(b):
            cos_angle[i][labels[i]] = cos_angle[i][labels[i]] - self.m
        weighted_cos_angle = self.s * cos_angle
        log_probs = self.CrossEntropy(weighted_cos_angle, labels, use_label_smoothing=True)
        return log_probs


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
