import pytest
import torch
from residual.utils import Trainer

def test_trainer():
    config = [
      ResidualConfig(channels=64, conv_kernel_size=3, skip_kernel_size=1),
    ]

    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss()
    Trainer(
            model_config=config,
            n_classes=10,
            optimizer='Adam',
            criterion=torch.nn.CrossEntropyLoss()
        )


