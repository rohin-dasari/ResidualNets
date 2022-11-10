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
                        channels=64,
                        conv_kernel_size=3,
                        skip_kernel_size=1
                    )

            ]
    model = ResNet(data_shape, 10, configs=layers)
    data = torch.randn(data_shape)
    out = model.forward(data)
    import pdb; pdb.set_trace()




