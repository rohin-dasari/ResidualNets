from dataclasses import dataclass
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
    n_blocks: int = 2


class ResidualBlock(nn.Module):
    def __init__(self, in_size, channels, conv_kernel_size, skip_kernel_size=None, stride=1, device='cpu'):
        super().__init__()

        in_channels = in_size[1]
        dims = in_size[-1]
        main_padding = self.compute_padding(dims, conv_kernel_size, stride)
        self.main_block = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=conv_kernel_size, stride=stride, bias=False, padding=main_padding).to(device),
                nn.BatchNorm2d(channels).to(device),
                nn.ReLU().to(device),
                nn.Conv2d(channels, channels, kernel_size=conv_kernel_size, stride=stride, bias=False, padding=main_padding).to(device),
                nn.BatchNorm2d(channels).to(device)
            )

        
        skip_modules = []
        self.expansion = channels // in_channels
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
                        ).to(device)
                    )
            skip_modules.append(nn.BatchNorm2d(channels).to(device))
            self.expansion = 1
        self.skip_block = nn.Sequential(*skip_modules)


    def compute_padding(self, dims, kernel_size, stride):
        assert kernel_size % 2 == 1, "kernel size must be an odd number"
        return (stride*(dims - 1) + kernel_size - dims) // 2

    def forward(self, x):
        f = self.main_block(x)
        s = self.skip_block(x)
        s = s.repeat(1, self.expansion, 1, 1)
        return F.relu(f + s)


class ResidualLayer(nn.Module):
    """
    A residual layer consists of two residual blocks with the same kernel sizes
    """
    #def __init__(self, in_size, channels, conv_kernel_size, skip_kernel_size=None, stride=1):
    def __init__(self, config: ResidualConfig, device='cpu'):
        super().__init__()
        assert config.n_blocks > 0, "must have at least one residual block in a residual layer"
        self.channels = config.channels
        block1 = ResidualBlock(
                    config.in_size,
                    config.channels,
                    config.conv_kernel_size,
                    config.skip_kernel_size,
                    config.stride,
                    device=device
                )

        in_size = (config.in_size[0], config.channels, config.in_size[2], config.in_size[3])
        blocks = [block1]
        for block in range(config.n_blocks-1):
            blocks.append(
                    ResidualBlock(
                        in_size,
                        config.channels,
                        config.conv_kernel_size,
                        config.skip_kernel_size,
                        config.stride,
                        device=device
                    )
                )
        self.blocks = nn.Sequential(*blocks)

        
    def forward(self, x):
        return self.blocks(x)

