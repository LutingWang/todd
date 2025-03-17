__all__ = [
    'SwiGLU',
    'ApproximateGELU',
]

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):

    def __init__(
        self,
        *args,
        in_features: int,
        hidden_features: int,
        out_features: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._linear1 = nn.Linear(in_features, hidden_features)
        self._linear2 = nn.Linear(in_features, hidden_features)
        self._norm = nn.LayerNorm(hidden_features, 1e-6)
        self._projector = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self._linear1(x)) * self._linear2(x)
        x = self._norm(x)
        x = self._projector(x)
        return x


class ApproximateGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
