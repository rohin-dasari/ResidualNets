import numpy as np
from torch import nn
from torch.nn import functional as F
from .residual_block import ResidualLayer


class ResNet(nn.Module):
    def __init__(self, in_size, out_size, configs, average_pool_kernel_size=1):
        super().__init__()
        in_channels = in_size[1]
        self.head = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
        self.out_size = out_size
        self.configs = configs
        self.average_pool_kernel_size = average_pool_kernel_size
    
    def make_body(self, in_size):
        self.body = nn.Sequential()
        for i, config in enumerate(self.configs):
            config.in_size = in_size
            self.body.add_module(f'body_{i}', ResidualLayer(config))
            in_size = (in_size[0], config.channels, in_size[2], in_size[3])
    
    def make_tail(self):
        last_body_piece = list(self.body.children())[-1]
        channel_size = last_body_piece.channels
        pool_kernel = (self.average_pool_kernel_size, self.average_pool_kernel_size)
        self.tail = nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_kernel),
                    nn.Flatten(),
                    nn.Linear(channel_size, self.out_size)
                )
        #self.tail.add_module('avg_pool', nn.AdaptiveAvgPool2d(pool_kernel))


    def forward(self, x):
        x = self.head(x)
        # lazily create the body of the model
        if not hasattr(self, 'body'):
            self.make_body(x.shape)

        x = self.body(x)


        # lazily create the tail of the model
        if not hasattr(self, 'tail'):
            self.make_tail()
            
        return self.tail(x)

