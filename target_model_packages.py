from __future__ import print_function

import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple


class Classifier_4(nn.Module):
    def __init__(self, nc, ndf, nz, img_size, block_idx=0):
        super(Classifier_4, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        self.img_size = img_size

        self.block1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
            # state size. (ndf*8) x 4 x 4
        )

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * int((self.img_size/16)) * int((self.img_size/16)), nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def _record_time_memory(self, block, x, device):
        if device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            x = block(x)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)  # Milliseconds
            memory_usage = torch.cuda.memory_allocated() / (1024 ** 2)  # Megabytes

        else:
            # CPU Time and Memory
            start_time = time.time()  # Record CPU start time
            memory_usage_before = psutil.Process().memory_info().rss / (1024 ** 2)  # Memory before in MB

            x = block(x)

            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            memory_usage_after = psutil.Process().memory_info().rss / (1024 ** 2)  # Memory after in MB
            memory_usage = memory_usage_after - memory_usage_before  # Memory difference in MB

        return x, elapsed_time, memory_usage

    def forward(self, x, block_idx=None, record_time_memory=False, release=False, device="cuda"):
        times, feat, memories = [], [], []
        blocks = [self.block1, self.block2, self.block3, self.block4]

        x = x.view(-1, self.nc, self.img_size, self.img_size)

        for idx, block in enumerate(blocks):
            if record_time_memory:
                x, elapsed_time, memory_usage = self._record_time_memory(block, x, device)
                times.append(elapsed_time)
                memories.append(memory_usage)
            else:
                x = block(x)

            feat.append(x)

        # x = self.block1(x)
        # feat.append(x)
        # x = self.block2(x)
        # feat.append(x)
        # x = self.block3(x)
        # feat.append(x)
        # x = self.block4(x)
        # feat.append(x)
        x = x.view(-1, self.ndf * 8 * int((self.img_size/16)) * int((self.img_size/16)))
        # out = self.fc(x)

        if record_time_memory:
            out, elapsed_time, memory_usage = self._record_time_memory(self.fc, x, device)
            times.append(elapsed_time)
            memories.append(memory_usage)
        else:
            out = self.fc(x)

        if record_time_memory:
            return feat, times, memories
        else:
            if release:
                return feat, F.softmax(out, dim=1)
            else:
                return feat, F.log_softmax(out, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, nc=3, img_size=64):
        super(ResNet, self).__init__()
        self.inchannel = img_size
        self.mid = int(img_size / 64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512 * (self.mid*2) * (self.mid*2), num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x, block_idx=None, release=False):
        feat = []
        out = self.conv1(x)
        feat.append(out)
        out = self.layer1[0](out)
        feat.append(out)
        for layer in self.layer1[1:]:
            out = layer(out)
            feat.append(out)
        out = self.layer2[0](out)
        feat.append(out)
        for layer in self.layer2[1:]:
            out = layer(out)
            feat.append(out)
        out = self.layer3[0](out)
        feat.append(out)
        for layer in self.layer3[1:]:
            out = layer(out)
            feat.append(out)
        out = self.layer4[0](out)
        feat.append(out)
        for layer in self.layer4[1:]:
            out = layer(out)
            feat.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feat.append(out)
        out = self.fc(out)
        if release:
            return feat, F.softmax(out, dim=1)
        else:
            return feat, F.log_softmax(out, dim=1)


def rn18(num_classes, nc, img_size):
    return ResNet(ResidualBlock, num_classes, nc, img_size)


class VGG(nn.Module):

    def __init__(self, features, num_class=10, img_size=64):
        super().__init__()
        self.features = features
        self.img_size = img_size
        self.mid = int(img_size / 64)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 *2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, num_class)
        )

    def forward(self, x, block_idx=None, release=False):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            features.append(x)

        # x = self.avgpool(x)
        output = x.view(x.size()[0], -1)
        output = self.classifier(output)

        if release:
            return features, F.softmax(output, dim=1)
        else:
            return features, F.log_softmax(output, dim=1)


def make_layers(nc, cfg, batch_norm=False):
    layers = []

    input_channel = nc
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg16_bn(nc, nz, img_size):
    nc = nc
    nz = nz
    img_size = img_size
    return VGG(make_layers(nc, cfg['D'], batch_norm=True), nz, img_size)

def vgg19(nc, nz, img_size):
    nc = nc
    nz = nz
    img_size = img_size
    return VGG(make_layers(nc, cfg['E'], batch_norm=True), nz, img_size)

def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


# class EncoderWithResidualForIR152(nn.Module):
#     def __init__(self, ndf):
#         super(EncoderWithResidualForIR152, self).__init__()
#         self.conv2ed = nn.Conv2d(ndf, ndf//4, 3, 1, 1)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(ndf, ndf//4, 3, 1, 1),
#             nn.SiLU(),
#             nn.Conv2d(ndf//4, ndf//4, 3, 1, 1),
#         )
#         self.res_edge = nn.Sequential(
#             nn.Conv2d(ndf//4, 2, 3, 1, 1),
#             nn.BatchNorm2d(2)
#         )
#
#
#     def forward(self, x):
#         first_conv_out = self.conv2ed(x)
#
#         encoder_out = self.encoder(x)
#
#         combined_out = first_conv_out + encoder_out
#
#         final_out = F.relu(self.res_edge(combined_out))
#
#         return final_out
#
#
# class DecoderWithResidualForIR152(nn.Module):
#     def __init__(self, ndf):
#         super(DecoderWithResidualForIR152, self).__init__()
#         self.conv2ed = nn.Conv2d(2, ndf//4, 3, 1, 1)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(2, ndf//4, 3, 1, 1),
#             nn.SiLU(),
#             nn.Conv2d(ndf//4, ndf//4, 3, 1, 1),
#         )
#         self.res_cloud = nn.Sequential(
#             nn.Conv2d(ndf//4, ndf, 3, 1, 1),
#             nn.BatchNorm2d(ndf),
#         )
#
#     def forward(self, x):
#         first_conv_out = self.conv2ed(x)
#
#         encoder_out = self.encoder(x)
#
#         combined_out = first_conv_out + encoder_out
#
#         final_out = F.relu(self.res_cloud(combined_out))
#
#         return final_out
# class EncoderWithResidualForIR152(nn.Module):
#     def __init__(self, ndf):
#         super(EncoderWithResidualForIR152, self).__init__()
#         self.conv2ed = nn.Conv2d(ndf, ndf//4, 3, 1, 1)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(ndf, ndf//4, 3, 1, 1),
#             nn.SiLU(),
#             nn.Conv2d(ndf//4, ndf//4, 3, 1, 1),
#         )
#         self.res_edge = nn.Sequential(
#             nn.Conv2d(ndf//4, 2, 3, 1, 1),
#             nn.BatchNorm2d(2)
#         )
#
#
#     def forward(self, x):
#         first_conv_out = self.conv2ed(x)
#
#         encoder_out = self.encoder(x)
#
#         combined_out = first_conv_out + encoder_out
#
#         final_out = F.relu(self.res_edge(combined_out))
#
#         return final_out
#
#
# class DecoderWithResidualForIR152(nn.Module):
#     def __init__(self, ndf):
#         super(DecoderWithResidualForIR152, self).__init__()
#         self.conv2ed = nn.Conv2d(ndf, ndf*4, 3, 1, 1)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(ndf, ndf*4, 3, 1, 1),
#             nn.SiLU(),
#             nn.Conv2d(ndf*4, ndf*4, 3, 1, 1),
#         )
#         self.res_cloud = nn.Sequential(
#             nn.Conv2d(ndf*4, ndf * 8, 3, 1, 1),
#             nn.BatchNorm2d(ndf * 8),
#         )
#
#     def forward(self, x):
#         first_conv_out = self.conv2ed(x)
#
#         encoder_out = self.encoder(x)
#
#         combined_out = first_conv_out + encoder_out
#
#         final_out = F.relu(self.res_cloud(combined_out))
#
#         return final_out


# class Flatten(Module):
#     def forward(self, input):
#         return input.view(input.size(0), -1)
#
#
# class Backbone64_AE(Module):
#     def __init__(self, input_size, num_layers, nc, mode='ir_se'):
#         super(Backbone64_AE, self).__init__()
#         assert input_size[0] in [64, 128], "input_size should be [112, 112] or [224, 224]"
#         assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
#         assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
#         blocks = get_blocks(num_layers)
#         if mode == 'ir':
#             unit_module = bottleneck_IR
#         elif mode == 'ir_se':
#             unit_module = bottleneck_IR_SE
#         self.input_layer = Sequential(Conv2d(nc, 64, (3, 3), 1, 1, bias=False),
#                                       BatchNorm2d(64),
#                                       PReLU(64))
#
#         self.output_layer = Sequential(BatchNorm2d(512),
#                                        Dropout(),
#                                        Flatten(),
#                                        Linear(512 * 14 * 14, 10),
#                                        BatchNorm1d(512))
#
#         modules = []
#         for block in blocks:
#             for bottleneck in block:
#                 modules.append(
#                     unit_module(bottleneck.in_channel,
#                                 bottleneck.depth,
#                                 bottleneck.stride))
#             if bottleneck.depth == 128:
#                 modules.append(EncoderWithResidualForIR152(128))
#                 modules.append(DecoderWithResidualForIR152(128))
#
#         self.body = Sequential(*modules)
#
#         self._initialize_weights()
#
#     def forward(self, x):
#         features = []
#         x = self.input_layer(x)
#         for layer in self.body:
#             x = layer(x)
#             features.append(x)
#         return features
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
# def get_blocks(num_layers):
#     if num_layers == 50:
#         blocks = [
#             get_block(in_channel=64, depth=64, num_units=3),
#             get_block(in_channel=64, depth=128, num_units=4),
#             get_block(in_channel=128, depth=256, num_units=14),
#             get_block(in_channel=256, depth=512, num_units=3)
#         ]
#     elif num_layers == 100:
#         blocks = [
#             get_block(in_channel=64, depth=64, num_units=3),
#             get_block(in_channel=64, depth=128, num_units=13),
#             get_block(in_channel=128, depth=256, num_units=30),
#             get_block(in_channel=256, depth=512, num_units=3)
#         ]
#     elif num_layers == 152:
#         blocks = [
#             get_block(in_channel=64, depth=64, num_units=3),
#             get_block(in_channel=64, depth=128, num_units=8),
#             get_block(in_channel=128, depth=256, num_units=36),
#             get_block(in_channel=256, depth=512, num_units=3)
#         ]
#
#     return blocks
#
#
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Backbone64(Module):
    def __init__(self, input_size, num_layers, nc, mode='ir'):
        super(Backbone64, self).__init__()
        assert input_size[0] in [64, 128], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(),
                                       Flatten(),
                                       Linear(512 * 14 * 14, 10),
                                       BatchNorm1d(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def forward(self, x):
        features = []
        x = self.input_layer(x)
        features.append(x)
        for layer in self.body:
            x = layer(x)
            features.append(x)
        return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class IR152(nn.Module):
    def __init__(self, num_classes=10, nc=3, img_size=64):
        super(IR152, self).__init__()
        self.feature = IR_152_64((img_size, img_size), nc)
        self.feat_dim = 512
        self.num_classes = num_classes
        self.mid = int(img_size / 16)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(),
                                          Flatten(),
                                          nn.Linear(512 * self.mid * self.mid, 512),
                                          nn.BatchNorm1d(512))

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x, block_idx=None, release=False):
        feat = self.feature(x)
        feat_out = feat[-1]
        feat_out = self.output_layer(feat_out)
        feat_out = feat_out.view(feat_out.size(0), -1)
        out = self.fc_layer(feat_out)
        if release:
            return feat, F.softmax(out, dim=1)
        else:
            return feat, F.log_softmax(out, dim=1)


def IR_152_64(input_size, nc):
    """Constructs a ir-152 model.
    """
    model = Backbone64(input_size, 152, nc, 'ir')

    return model



class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] -> [B, embed_dim, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=16, in_channels=3, num_classes=10,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, block_idx=None, release=False):
        feat = []
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
            feat.append(x.clone())

        x = self.norm(x)
        out = self.head(x[:, 0])

        if release:
            return feat, F.softmax(out, dim=1)
        else:
            return feat, F.log_softmax(out, dim=1)