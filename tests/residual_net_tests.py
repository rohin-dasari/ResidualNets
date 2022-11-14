import pytest
import torch
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


def test_resnet18():
    resnet18_config = [
      ResidualConfig(channels=64, conv_kernel_size=3, skip_kernel_size=1),
      ResidualConfig(channels=128, conv_kernel_size=3, skip_kernel_size=1, stride=2, n_blocks=4),
      ResidualConfig(channels=256, conv_kernel_size=3, skip_kernel_size=1),
      ResidualConfig(channels=256, conv_kernel_size=3, skip_kernel_size=1),
      ResidualConfig(channels=512, conv_kernel_size=3, skip_kernel_size=1)
    ]

    data_shape = (1, 3, 100, 100)
    data = torch.randn(data_shape)
    out_shape = 10

    resnet18 = ResNet(data_shape, out_shape, configs=resnet18_config)
    resnet18.forward(data)


def test_model_summary():
    resnet18_config = [
      ResidualConfig(channels=64, conv_kernel_size=3, skip_kernel_size=1),
    ]
    data_shape = (1, 3, 100, 100)
    data = torch.randn(data_shape)
    out_shape = 10

    model = ResNet(data_shape, out_shape, configs=resnet18_config)
    print(model.get_parameter_count())
    model.summary()



