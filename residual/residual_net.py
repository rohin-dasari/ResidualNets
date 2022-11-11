import numpy as np
from torch import nn
from torch.nn import functional as F
from .residual_block import ResidualLayer


class ResNet(nn.Module):
    def __init__(self, in_size, out_size, configs, average_pool_kernel_size=1, device='cpu'):
        super().__init__()
        self.in_size = in_size
        in_channels = self.in_size[1]
        self.head = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3).to(device),
                nn.BatchNorm2d(64).to(device),
                nn.ReLU().to(device),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)
                )
        self.out_size = out_size
        self.configs = configs
        self.average_pool_kernel_size = average_pool_kernel_size
        self.device = device
    
    def make_body(self, in_size):
        self.body = nn.Sequential()
        in_channels = in_size[1]
        for i, config in enumerate(self.configs):
            config.in_size = (in_size[0], in_channels, in_size[2], in_size[3])
            self.body.add_module(f'body_{i}', ResidualLayer(config, device=self.device))
            in_channels = config.channels
    
    def make_tail(self):
        last_body_piece = list(self.body.children())[-1]
        channel_size = last_body_piece.channels
        pool_kernel = (self.average_pool_kernel_size, self.average_pool_kernel_size)
        self.tail = nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_kernel).to(self.device),
                    nn.Flatten().to(self.device),
                    nn.Linear(channel_size, self.out_size).to(self.device)
                )


    def compile(self):
        """
        pass some sample data through the model to build all the models components
        """
        data = torch.randn(self.in_size).to(self.device)
        self.forward(data)
        assert hasattr(self, 'body')
        assert hasattr(self, 'tail')

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

