import pytest
import torch
import numpy as np

from residual.residual_block import ResidualConfig, ResidualLayer, ResidualBlock


def test_shape_consistency():
    data_shape = (1, 64 ,13, 13)
    

    config = ResidualConfig(
            in_size=data_shape,
            channels=128,
            conv_kernel_size=3,
            skip_kernel_size=1
        )

    layer = ResidualLayer(config)
    data = torch.randn(data_shape)
    out = layer.forward(data)






