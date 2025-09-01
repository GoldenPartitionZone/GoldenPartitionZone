from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Inversion_4(nn.Module):
    def __init__(self, nc, ngf, ndf, nz, img_size, block_idx):
        super(Inversion_4, self).__init__()

        self.nc = nc        # Number of input image channels
        self.ngf = ngf      # Number of generator feature maps
        self.ndf = ndf      # Number of encoder feature maps (same as Classifier_4)
        self.nz = nz        # Latent vector dimension
        self.img_size = img_size  # Image size (e.g., 64x64)
        self.block_idx = block_idx  # Which block's output to reconstruct
        self.spatial_size = 64

        # Adjust the input channels and feature map size based on the selected block
        if block_idx == 4:
            input_channels = ndf * 8  # block4 output (ndf*8) x 4 x 4
            self.spatial_size = img_size // 16
            self.decoder = self._create_decoder(input_channels, 4)  # 4 transpose conv layers
        elif block_idx == 3:
            input_channels = ndf * 4  # block3 output (ndf*4) x 8 x 8
            self.spatial_size = img_size // 8
            self.decoder = self._create_decoder(input_channels, 3)  # 3 transpose conv layers
        elif block_idx == 2:
            input_channels = ndf * 2  # block2 output (ndf*2) x 16 x 16
            self.spatial_size = img_size // 4
            self.decoder = self._create_decoder(input_channels, 2)  # At least 2 transpose conv layers
        elif block_idx == 1:
            input_channels = ndf      # block1 output (ndf) x 32 x 32
            self.spatial_size = img_size // 2
            self.decoder = self._create_decoder(input_channels, 1)  # At least 1 transpose conv layer
        elif block_idx == 5:  # If it's from the fully connected layer
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),  # Adjusted according to your request
                nn.BatchNorm2d(ngf * 8),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
                nn.BatchNorm2d(ngf * 4),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
                nn.BatchNorm2d(ngf * 2),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
                nn.BatchNorm2d(ngf),
                nn.Tanh(),
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
                nn.Sigmoid()  # Final output size (nc) x 64 x 64
            )
        else:
            raise ValueError("block_idx must be between 1 and 4, or 'fc'")

    def _create_decoder(self, input_channels, num_layers):
        # Dynamically create the decoder based on the number of layers and the block selected
        layers = []
        ngf = input_channels // 2
        for i in range(num_layers, 0, -1):
            if self.spatial_size == self.img_size // 2:
                layers.append(nn.ConvTranspose2d(input_channels, self.nc, 4, 2, 1))
                layers.append(nn.Sigmoid())  # Output image size is (nc) x 64 x 64
                break
            else:
                layers.append(nn.ConvTranspose2d(input_channels, ngf, 4, 2, 1))
                layers.append(nn.BatchNorm2d(ngf))
                layers.append(nn.Tanh())  # ReLU replaced by Tanh
                input_channels = ngf
                ngf = ngf // 2
                self.spatial_size *= 2

        return nn.Sequential(*layers)

    def forward(self, x):
        # Assuming x is the output from a block, its shape matches the block output
        if self.block_idx == 5:
            x = x.view(-1, self.nz, 1, 1)
            x = self.decoder(x)
            x = x.view(-1, self.nc, self.img_size, self.img_size)  # Reshape back to image size
        else:
            x = self.decoder(x)
        return x


class InversionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(InversionResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, output_padding=0,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0,
                               bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class InversionResNet(nn.Module):
    def __init__(self, InversionResidualBlock, block_idx, num_classes=10, nc=3, img_size=64):
        super(InversionResNet, self).__init__()
        self.img_size = img_size
        self.block_idx = block_idx
        self.ndf = 64  # Number of feature maps in the first convolutional layer

        self.deconv_layers = []

        # Define residual blocks based on block_idx
        if block_idx <= 3:
            # Block 0-14
            if block_idx == 3 or 2:
                self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        elif block_idx in range(4, 6):
            # Block 15-28
            if block_idx == 5:
                self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 64, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        elif block_idx in range(6, 8):
            # Block 29-42
            if block_idx == 7:
                self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 128, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 64, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        elif block_idx in range(8, 11):
            # Block 43-57
            if block_idx == 9 or 10:
                self.deconv_layers.append(InversionResidualBlock(512, 512, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(512, 512, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(512, 256, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 256, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(256, 128, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(128, 128, stride=1, kernel_size=3))
            self.deconv_layers.append(InversionResidualBlock(128, 64, stride=2, kernel_size=4))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            self.deconv_layers.append(InversionResidualBlock(64, 64, stride=1))
            final_channels = 64
        else:
            raise ValueError("block_idx out of range")

        # Convert list to Sequential
        self.deconv_layers = nn.Sequential(*self.deconv_layers)

        # Define final layers to get to the original image size
        self.final_layers = nn.Sequential(
            nn.ConvTranspose2d(final_channels, nc, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layers(x)
        return x


def inversion_resnet(block_idx, num_classes, nc, img_size):
    return InversionResNet(InversionResidualBlock, block_idx, num_classes, nc, img_size)


class inversion_vgg(nn.Module):
    def __init__(self, nc, ngf, ndf, nz, img_size, block_idx):
        super(inversion_vgg, self).__init__()

        self.nc = nc        # Number of input image channels
        self.ngf = ngf      # Number of generator feature maps
        self.ndf = ndf      # Number of encoder feature maps (same as Classifier_4)
        self.nz = nz        # Latent vector dimension
        self.img_size = img_size  # Image size (e.g., 64x64)
        self.block_idx = block_idx  # Which block's output to reconstruct
        self.spatial_size = 64

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, nc, 3, 1, 1),
            nn.Sigmoid()  # Final output size (nc) x 64 x 64
        )


    def forward(self, x):
        # Assuming x is the output from a block, its shape matches the block output
        x = self.decoder(x)
        return x


class inversion_vgg_pv4(nn.Module):
    def __init__(self, nc, ngf, ndf, nz, img_size, block_idx):
        super(inversion_vgg_pv4, self).__init__()

        self.nc = nc        # Number of input image channels
        self.ngf = ngf      # Number of generator feature maps
        self.ndf = ndf      # Number of encoder feature maps (same as Classifier_4)
        self.nz = nz        # Latent vector dimension
        self.img_size = img_size  # Image size (e.g., 64x64)
        self.block_idx = block_idx  # Which block's output to reconstruct
        self.spatial_size = 64

        self.decoderx = nn.Sequential(
            nn.ConvTranspose2d(12, 12, 3, 1, 1),  # Adjusted according to your request
            nn.BatchNorm2d(12),
            nn.Tanh(),
            nn.ConvTranspose2d(12, 12, 3, 1, 1),
            nn.Tanh(),
            nn.ConvTranspose2d(12, 32, 3, 1, 1),
            nn.ConvTranspose2d(32, 128, 4, 2, 1),  # Adjusted according to your request
        )

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(8, 64, 4, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, nc, 3, 1, 1),
            nn.Sigmoid()  # Final output size (nc) x 64 x 64
        )


    def forward(self, x):
        # Assuming x is the output from a block, its shape matches the block output
        x = self.decoderx(x)
        x = self.decoder(x)
        return x




class bottleneck_IR_reverse(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(bottleneck_IR_reverse, self).__init__()

        if stride == 2:
            self.shortcut_layer = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False)

        else:
            self.shortcut_layer = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=1, stride=stride, output_padding=stride-1, bias=False),
                nn.BatchNorm2d(out_channel),
            )

        if stride == 1:
            self.res_layer = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PReLU(out_channel),
                nn.ConvTranspose2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, output_padding=stride-1, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        elif stride == 2:
            self.res_layer = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PReLU(out_channel),
                nn.ConvTranspose2d(out_channel, out_channel, kernel_size=4, stride=stride, padding=1,
                                   output_padding=0, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class inversion_ir152(nn.Module):
    def __init__(self, nc, ngf, ndf, nz, img_size, block_idx):
        super(inversion_ir152, self).__init__()

        self.nc = nc        # Number of input image channels
        self.ngf = ngf      # Number of generator feature maps
        self.ndf = ndf      # Number of encoder feature maps (same as Classifier_4)
        self.nz = nz        # Latent vector dimension
        self.img_size = img_size  # Image size (e.g., 64x64)
        self.block_idx = block_idx  # Which block's output to reconstruct
        self.spatial_size = 64

        self.deconv_layers = []
        self.final_conv = nn.Sequential()

        if block_idx <= 3:
            # Block 0-14
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=2))
            final_channels = 64
        elif block_idx in range(4, 7):
            # Block 15-28
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 64, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=2))
            final_channels = 64
        elif block_idx in range(7, 9):
            # Block 29-42
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 64, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=2))
            final_channels = 64
        elif block_idx in range(9, 12):
            # Block 43-57
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 64, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=2))
            final_channels = 64
        elif block_idx in range(12, 13):
            # Block 43-57
            self.deconv_layers.append(bottleneck_IR_reverse(256, 128, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 64, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=2))
            final_channels = 64
        elif block_idx in range(25, 40):
            # Block 43-57
            for cnt_layer in range(10):
                self.deconv_layers.append(bottleneck_IR_reverse(256, 256, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(256, 128, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 64, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=2))
            # self.deconv_layers.append(bottleneck_IR_reverse(256, 128, stride=2))
            # self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(128, 64, stride=2))
            # self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=2))
            final_channels = 64
        elif block_idx == 47:
            # Block 43-57
            for cnt_layer in range(35):
                self.deconv_layers.append(bottleneck_IR_reverse(256, 256, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(256, 128, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 64, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=2))
            final_channels = 64
        elif block_idx == 50:
            # Block 43-57
            self.deconv_layers.append(bottleneck_IR_reverse(512, 256, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(256, 128, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(128, 64, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=2))
            self.deconv_layers.append(bottleneck_IR_reverse(64, 3, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(512, 256, stride=2))
            # for cnt_layer in range(2):
            #     self.deconv_layers.append(bottleneck_IR_reverse(256, 256, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(256, 128, stride=2))
            # for cnt_layer in range(4):
            #     self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(128, 64, stride=2))
            # self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
            # self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=2))
            # final_channels = 64

        else:
            raise ValueError("block_idx out of range")

        # Convert list to Sequential
        self.deconv_layers = nn.Sequential(*self.deconv_layers)

        # Define final layers to get to the original image size
        self.final_layers = nn.Sequential(
            nn.ConvTranspose2d(3, nc, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.deconv_layers(x)
        x = self.final_layers(x)
        # x = self.final_conv(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.size()
        assert C % self.groups == 0, "Channels must be divisible by groups"
        x = x.view(B, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.view(B, C, H, W)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.activation = nn.PReLU()

    def forward(self, x):
        out = self.block(x)
        out += x
        return self.activation(out)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(channels, num_heads=2, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)
        attn, _ = self.mha(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn
        x_flat = x_flat + self.ff(x_flat)
        out = x_flat.permute(0, 2, 1).view(b, c, h, w)
        return out


class EntropyEnhancerNet(nn.Module):
    def __init__(self, in_channels=512, out_channels=1024, num_blocks=1):
        super(EntropyEnhancerNet, self).__init__()

        # Module 1: Feature Extractor
        self.feature_extractor = nn.Sequential(
            *[ResidualBlock(in_channels) for _ in range(num_blocks)]
        )

        # Module 2: Feature Expansion
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

        # Module 3: Diversification
        self.diversify = nn.Sequential(
            SEBlock(out_channels),
            # ChannelShuffle(groups=4),
            nn.GroupNorm(num_groups=8, num_channels=out_channels)
        )

        # Module 4: Output Refinement
        self.final = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.expansion(out)
        out = self.diversify(out)
        out = self.final(out)
        return out


class InversionDecoder(nn.Module):
    def __init__(self, in_channels=1024, image_channels=3):
        super(InversionDecoder, self).__init__()
        self.deconv_layers=[]

        self.deconv_layers.append(bottleneck_IR_reverse(1024, 512, stride=1))
        self.deconv_layers.append(bottleneck_IR_reverse(512, 256, stride=2))
        self.deconv_layers.append(bottleneck_IR_reverse(256, 256, stride=1))
        self.deconv_layers.append(bottleneck_IR_reverse(256, 128, stride=2))
        self.deconv_layers.append(bottleneck_IR_reverse(128, 128, stride=1))
        self.deconv_layers.append(bottleneck_IR_reverse(128, 64, stride=2))
        self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=1))
        self.deconv_layers.append(bottleneck_IR_reverse(64, 64, stride=2))
        final_channels = 64

        self.decode = nn.Sequential(*self.deconv_layers)

    def forward(self, z):
        return self.decode(z)


class InversionModel(nn.Module):
    def __init__(self, feature_channels=512, enhanced_channels=1024, image_channels=3):
        super(InversionModel, self).__init__()
        self.een = EntropyEnhancerNet(in_channels=feature_channels, out_channels=enhanced_channels)
        self.decoder = InversionDecoder(in_channels=enhanced_channels, image_channels=image_channels)
        self.final_layers = nn.Sequential(
            nn.ConvTranspose2d(64, image_channels, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, z):
        z_enh = self.een(z)
        x_recon = self.decoder(z_enh)
        x_recon = self.final_layers(x_recon)
        return x_recon, z_enh

def distance_correlation(x, z):
    """
    Compute the distance correlation between two tensors x and z of shape:
    - x: (B, C1, H1, W1) or (B, D1)
    - z: (B, C2, H2, W2) or (B, D2)

    Returns:
        Scalar tensor: distance correlation value ∈ [0, 1]
    """
    # Flatten to (B, D)
    x = x.view(x.size(0), -1)
    z = z.view(z.size(0), -1)

    def pairwise_distances(a):
        # Euclidean pairwise distance matrix: (B, B)
        return torch.cdist(a, a, p=2)

    A = pairwise_distances(x)
    B = pairwise_distances(z)

    # Double centering
    A_mean = A - A.mean(dim=0, keepdim=True) - A.mean(dim=1, keepdim=True) + A.mean()
    B_mean = B - B.mean(dim=0, keepdim=True) - B.mean(dim=1, keepdim=True) + B.mean()

    dcov = (A_mean * B_mean).mean()
    dvar_x = (A_mean * A_mean).mean()
    dvar_z = (B_mean * B_mean).mean()

    return dcov / (torch.sqrt(dvar_x * dvar_z) + 1e-8)


def sample_wise_spread(z):
    """
    输入: z ∈ [B, C, H, W]
    输出: z_enhanced ∈ [B, C, H, W]，增强后的特征
    目标: 每个样本间距离尽量远
    """
    B, C, H, W = z.shape
    z_flat = z.view(B, -1)  # B x (C*H*W)

    # 1. 中心化处理（去掉偏移）
    z_mean = z_flat.mean(dim=0, keepdim=True)
    z_centered = z_flat - z_mean  # B x D

    # 2. 计算样本协方差矩阵（B x B）
    sim_matrix = torch.mm(z_centered, z_centered.T)  # inner product
    norms = torch.norm(z_centered, dim=1, keepdim=True)
    cosine_sim = sim_matrix / (norms @ norms.T + 1e-8)

    # 3. 负梯度方向：远离相似样本（近似反正交化）
    penalty = cosine_sim.mean(dim=1, keepdim=True)  # B x 1
    z_spread = z_centered * (1 + penalty)  # 放大相似样本间距

    # 4. reshape 回原始
    return z_spread.view(B, C, H, W)


def energy_cap(z, max_norm=1.0):
    norm = z.view(z.size(0), -1).norm(p=2, dim=1, keepdim=True)  # B x 1
    scale = torch.clamp(max_norm / (norm + 1e-6), max=1.0)
    return z * scale.view(-1, 1, 1, 1)


class AttentionAsConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels, bias=False
        )  # 深度卷积模拟 attention map
        self.pw_conv = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False
        )  # 点卷积整合通道信息
        self.act = nn.Sigmoid()  # 控制增强幅度

    def forward(self, x):
        attn = self.dw_conv(x)
        x = x * self.act(attn)  # 类似于 soft attention weighting
        return self.pw_conv(x)


class AAC_CA_Block(nn.Module):
    def __init__(self, channels, kernel_size=5, reduction=16):
        super().__init__()
        self.aac = AttentionAsConv(channels, kernel_size)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.aac(x)
        ca_weight = self.se(out)
        return out * ca_weight


class MSCA_Block(nn.Module):
    def __init__(self, channels):
        super(MSCA_Block, self).__init__()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)

    def forward(self, x):
        out = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        out = self.fuse(out)
        return out * x


class LSK_Block(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(LSK_Block, self).__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.pointwise(self.depthwise(x)))
        return x * attn


class FeatureEnhancer(nn.Module):
    def __init__(self, in_channels, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

    def forward(self, z):
        return self.net(z)


class DecoderWithAugmentation(nn.Module):
    def __init__(self, in_channels=512, image_channels=3):
        super().__init__()
        aug_channels = 512 + 512 + 1  # FFT + MaxMin + GradMag
        # self.enhancer = FeatureEnhancer(in_channels)

        # self.net = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(in_channels//4),
        #     nn.PReLU(),
        #     nn.ConvTranspose2d(in_channels//4, in_channels//2, kernel_size=4, stride=2, padding=1),  # 可选 residual 输出
        #     nn.ConvTranspose2d(in_channels // 2, in_channels, kernel_size=4, stride=2, padding=1),
        # )

        self.deconv_block1 = nn.Sequential(
            # nn.ConvTranspose2d(in_channels + in_channels//8, 256, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            # SelfAttention(256),
            # AAC_CA_Block(256),
            # AttentionAsConv(256),
        )

        self.deconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            # nn.Tanh(),
            nn.PReLU(),
            # SelfAttention(128),
            # AAC_CA_Block(128),
            # AttentionAsConv(128),
        )

        self.deconv_block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.PReLU(),
            # SelfAttention(64),
            # AAC_CA_Block(64),
            # LSK_Block(64),
            # AttentionAsConv(64),
        )

        self.deconv_block4 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            # nn.Tanh(),
            nn.PReLU(),
            # SelfAttention(64),
            # AAC_CA_Block(64),
            # MSCA_Block(64),
            # AttentionAsConv(64),
        )

        self.deconv_block5 = nn.Sequential(
            nn.ConvTranspose2d(64, image_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()  # 输出到 [0,1] 区间
        )

    def forward(self, z):
        # z_aug = self.net(z)              # (B, 1537, H, W)
        # z = torch.cat([z, z_aug], dim=1)
        # z_fused = self.augment_features(z)           # (B, 512, H, W)
        # out = self.deconv_blocks(z_aug)            # (B, 3, 64, 64)
        # return z_aug, out

        # z_global = F.adaptive_avg_pool2d(z, output_size=1).squeeze(-1).squeeze(-1)
        # z_enhanced = self.enhancer(z)
        out1 = self.deconv_block1(z)  # (B, 3, 64, 64)

        out2 = self.deconv_block2(out1)

        out3 = self.deconv_block3(out2)

        out4 = self.deconv_block4(out3)

        out = self.deconv_block5(out4)
        return out

    def augment_features(self, z):
        fft = torch.fft.fft2(z, norm='ortho')
        fft_energy = torch.log(1 + torch.abs(fft.real) + torch.abs(fft.imag))

        # max_vals = F.max_pool2d(z, kernel_size=3, stride=1, padding=1)
        # min_vals = -F.max_pool2d(-z, kernel_size=3, stride=1, padding=1)
        # max_min_diff = max_vals - min_vals
        #
        # grad_mag = torch.sqrt(z.pow(2).sum(dim=1, keepdim=True))

        # 标准化处理（防止过大或分布偏移）
        def normalize(feat):
            mean = feat.mean(dim=[1, 2, 3], keepdim=True)
            std = feat.std(dim=[1, 2, 3], keepdim=True)
            return (feat - mean) / (std + 1e-6)

        fft_energy = normalize(fft_energy)
        # max_min_diff = normalize(max_min_diff)
        # grad_mag = normalize(grad_mag)

        # return torch.cat([z, fft_energy], dim=1)
        return z + fft_energy


class FeatureExploder(nn.Module):
    def __init__(self, in_channels, token_dim=256, num_tokens=64):
        super().__init__()
        self.flatten = nn.Flatten(2)  # B, C, H*W
        self.proj = nn.Conv1d(in_channels, token_dim, kernel_size=1)
        self.cls_tokens = nn.Parameter(torch.randn(1, num_tokens, token_dim))

    def forward(self, z):
        B, C, H, W = z.size()
        z_flat = self.flatten(z).transpose(1, 2)  # B, HW, C
        z_embed = self.proj(z_flat.transpose(1, 2))  # B, token_dim, HW
        z_embed = z_embed.transpose(1, 2)  # B, HW, token_dim
        return torch.cat([self.cls_tokens.expand(B, -1, -1), z_embed], dim=1)  # B, N+HW, token_dim


class MixerBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(x)
        return x


class FeatureRecomposer(nn.Module):
    def __init__(self, token_dim, out_channels, out_size=64):
        super().__init__()
        self.proj = nn.Linear(token_dim, out_size * out_size * out_channels)
        self.out_size = out_size
        self.out_channels = out_channels

    def forward(self, x):
        # x: (B, N_tokens, token_dim)
        img = self.proj(x.mean(dim=1))  # 聚合重构
        return img.view(-1, self.out_channels, self.out_size, self.out_size)


class ExplodeDecoder(nn.Module):
    def __init__(self, in_channels=512, image_channels=3, token_dim=256):
        super().__init__()
        self.explode = FeatureExploder(in_channels, token_dim=token_dim)
        self.mixer = nn.Sequential(
            MixerBlock(token_dim),
            MixerBlock(token_dim),
            MixerBlock(token_dim)
        )
        self.recompose = FeatureRecomposer(token_dim, out_channels=image_channels)

    def forward(self, z):
        tokens = self.explode(z)
        fused = self.mixer(tokens)
        out = self.recompose(fused)
        return out


# CoordConv：为输入添加位置编码
class CoordConv(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        xx = torch.linspace(-1, 1, W).repeat(H, 1).unsqueeze(0).expand(B, -1, -1).unsqueeze(1).to(x.device)
        yy = torch.linspace(-1, 1, H).repeat(W, 1).t().unsqueeze(0).expand(B, -1, -1).unsqueeze(1).to(x.device)
        return torch.cat([x, xx, yy], dim=1)

# ModulatedConv2D：可学习调制特征通道
class ModulatedConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.style = nn.Linear(style_dim, in_ch)
        self.norm = nn.InstanceNorm2d(in_ch)

    def forward(self, x, style_vec):
        style = self.style(style_vec).view(x.size(0), x.size(1), 1, 1)
        x = self.norm(x) * style
        return self.conv(x)

# Progressive Block
class UpBlockMod(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.mod_conv1 = ModulatedConv2D(in_ch, out_ch, 3, style_dim)
        self.mod_conv2 = ModulatedConv2D(out_ch, out_ch, 3, style_dim)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x, style):
        x = self.up(x)
        x = F.gelu(self.norm(self.mod_conv1(x, style)))
        x = F.gelu(self.norm(self.mod_conv2(x, style)))
        return x




class Inversion(nn.Module):
    def __init__(self, nc, ngf, nz, truncation, c):
        super(Inversion, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.truncation = truncation
        self.c = c

        self.decoder = nn.Sequential(
            # input is Z
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        # topk, indices = torch.topk(x, self.truncation)
        # topk = torch.clamp(torch.log(topk), min=-1000) + self.c
        # topk_min = topk.min(1, keepdim=True)[0]
        # topk = topk + F.relu(-topk_min)
        # x = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, topk)

        x = x.view(-1, self.nz, 1, 1)
        # x = x.pow(1 / 12)
        x = self.decoder(x)
        # x = x.view(-1, 3, 64, 64)
        return x


class DecoderForIR152(nn.Module):
    def __init__(self, block_idx, in_channels=512, image_channels=3):
        super().__init__()

        self.block_idx = block_idx

        # def decode():
        #     return nn.Sequential(
        #         nn.ConvTranspose2d(in_channels, ndf * 2, 3, 1, 1),  # Adjusted according to your request
        #         nn.BatchNorm2d(ndf * 2),
        #         nn.Tanh(),
        #         nn.ConvTranspose2d(ndf * 2, ndf * 2, 3, 1, 1),
        #         nn.ConvTranspose2d(ndf * 2, ndf * 2, 3, 1, 1),
        #     )

        # 定义构建函数
        def deconv1():
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.Tanh()
            )

        def deconv2():
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(in_channels // 4),
                nn.PReLU()
            )

        def deconv3():
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels // 4, in_channels // 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(in_channels // 8),
                nn.PReLU()
            )

        def deconv4():
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels // 8, in_channels // 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(in_channels // 8),
                nn.PReLU()
            )

        def deconv6():
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels // 2, image_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Sigmoid()
            )

        def deconv7():
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels // 4, image_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Sigmoid()
            )

        def deconv8():
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels // 8, image_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Sigmoid()
            )

        # 定义 block_idx 对应的模块序列
        block_map = {
            3: [deconv1(), deconv6()],
            13: [deconv1(), deconv6()],
            8: [deconv1(), deconv2(), deconv7()],
            17: [deconv1(), deconv2(), deconv7()],
            25: [deconv1(), deconv2(), deconv3(), deconv8()],
            26: [deconv1(), deconv2(), deconv7()],
            30: [deconv1(), deconv2(), deconv3(), deconv8()],
            39: [deconv1(), deconv2(), deconv3(), deconv8()],
            40: [deconv1(), deconv2(), deconv3(), deconv8()],
            48: [deconv1(), deconv2(), deconv3(), deconv8()],
        }

        assert block_idx in block_map, f"Unsupported block_idx: {block_idx}"

        # 构建解码网络
        self.deconv = nn.Sequential(*block_map[block_idx])

    def forward(self, z):
        return self.deconv(z)



class InversionForVIT(nn.Module):
    def __init__(self, nc, ngf, nz, truncation, c):
        super(InversionForVIT, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.truncation = truncation
        self.c = c

        self.decoder = nn.Sequential(
            # input is Z
            nn.ConvTranspose2d(256 * 128, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        # topk, indices = torch.topk(x, self.truncation)
        # topk = torch.clamp(torch.log(topk), min=-1000) + self.c
        # topk_min = topk.min(1, keepdim=True)[0]
        # topk = topk + F.relu(-topk_min)
        # x = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, topk)

        x = x.view(x.size(0), -1, 1, 1)
        # x = self.pre_proj(x)
        # x = x.pow(1 / 12)
        x = self.decoder(x)
        # x = x.view(-1, 3, 64, 64)
        return x