"""MLP Classifier for centralized and federated learning on UNSW-NB15 / synthetic."""
import torch
import torch.nn as nn
from typing import List, Union


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 191,
        hidden_dims: Union[List[int], int] = (128, 64, 32),
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
