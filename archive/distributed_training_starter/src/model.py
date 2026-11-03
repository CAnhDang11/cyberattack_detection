"""MLP Classifier for Distributed Training (DDP)."""
import torch
import torch.nn as nn
from typing import Union, List


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 191,
        hidden_dim: Union[List[int], int] = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim, hidden_dim // 2]
        else:
            hidden_dims = list(hidden_dim)

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
