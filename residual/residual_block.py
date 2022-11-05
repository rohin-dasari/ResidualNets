import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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


class ResnetLayer(nn.Module):
    pass


# some basic tests
block = ResidualBlock((4, 3, 7, 7), channels=3, conv_kernel_size=1, skip_kernel_size=5)
x = torch.randn(4, 3, 7, 7)
block.forward(x)

