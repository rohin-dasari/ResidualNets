import pytest
import torch
import numpy as np

from residual.residual_block import ResidualConfig
from residual.residual_net import ResNet


def test_basic_functionality():
    # shape -> (batch size, channels, img height, img width)
    data_shape = (1, 3, 50, 50)
    layers = [
                ResidualConfig(
                        channels=64,
                        conv_kernel_size=3,
                        skip_kernel_size=1
                    ),

                ResidualConfig(
                        channels=128,
                        conv_kernel_size=3,
                        skip_kernel_size=1
                    )

            ]
    model = ResNet(data_shape, 10, configs=layers)
    data = torch.randn(data_shape)
    out = model.forward(data)


def resnet18_test():
    resnet18_config = [
      ResidualConfig(channels=64, conv_kernel_size=3, skip_kernel_size=1),
      ResidualConfig(channels=128, conv_kernel_size=3, skip_kernel_size=1, stride=2),
      ResidualConfig(channels=256, conv_kernel_size=3, skip_kernel_size=1),
      ResidualConfig(channels=256, conv_kernel_size=3, skip_kernel_size=1),
      ResidualConfig(channels=512, conv_kernel_size=3, skip_kernel_size=1)
    ]

    data_shape = (1, 3, 100, 100)
    data = torch.randn(data_shape)
    out_shape = 10

    resnet18 = ResNet(data_shape, out_shape, configs=resnet18_config)
    resnet18.forward(data)



