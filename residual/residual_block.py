from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ResidualConfig:
    channels: int
    conv_kernel_size: int
    in_size: tuple = None
    skip_kernel_size: int = None
    stride: int = 1


class ResidualBlock(nn.Module):
    def __init__(self, in_size, channels, conv_kernel_size, skip_kernel_size=None, stride=1):
        super().__init__()

        in_channels = in_size[1]
        dims = in_size[-1]
        main_padding = self.compute_padding(dims, conv_kernel_size, stride)
        self.main_block = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=conv_kernel_size, stride=stride, bias=False, padding=main_padding),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=conv_kernel_size, stride=stride, bias=False, padding=main_padding),
                nn.BatchNorm2d(channels)
            )

        
        skip_modules = []
        if skip_kernel_size:
            skip_padding = self.compute_padding(dims, skip_kernel_size, stride)
            skip_modules.append(
                    nn.Conv2d(
                        in_channels,
                        channels,
                        kernel_size=skip_kernel_size,
                        stride=stride,
                        bias=False,
                        padding=skip_padding
                        )
                    )
            skip_modules.append(nn.BatchNorm2d(channels))
        self.skip_block = nn.Sequential(*skip_modules)


    def compute_padding(self, dims, kernel_size, stride):
        return (stride*(dims - 1) + kernel_size - dims) // 2

    def forward(self, x):
        f = self.main_block(x)
        s = self.skip_block(x)
        return F.relu(f + s)


class ResidualLayer(nn.Module):
    """
    A residual layer consists of two residual blocks with the same kernel sizes
    """
    #def __init__(self, in_size, channels, conv_kernel_size, skip_kernel_size=None, stride=1):
    def __init__(self, config: ResidualConfig):
        super().__init__()
        self.channels = config.channels
        self.block1 = ResidualBlock(
                    config.in_size,
                    config.channels,
                    config.conv_kernel_size,
                    config.skip_kernel_size,
                    config.stride
                )
        config.in_size = (config.in_size[0], config.channels, config.in_size[2], config.in_size[3])
        self.block2 = ResidualBlock(
                    config.in_size,
                    config.channels,
                    config.conv_kernel_size,
                    config.skip_kernel_size,
                    config.stride
                )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


# some basic tests
#block = ResidualBlock((4, 3, 7, 7), channels=3, conv_kernel_size=1, skip_kernel_size=5)
#layer = ResidualLayer(in_size=(4, 3, 7, 7), channels=3, conv_kernel_size=1, skip_kernel_size=1)
#x = torch.randn(4, 3, 7, 7)
#layer.forward(x)

