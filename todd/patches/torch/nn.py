__all__ = [
    'ModuleList',
    'ModuleDict',
    'Sequential',
]

from typing import Any

from torch import nn


class ModuleList(nn.ModuleList):

    def forward(self, *args, **kwargs) -> list[nn.Module]:
        return [m(*args, **kwargs) for m in self]


class ModuleDict(nn.ModuleDict):

    def forward(self, *args, **kwargs) -> dict[str, nn.Module]:
        return {k: m(*args, **kwargs) for k, m in self.items()}


class Sequential(nn.Sequential):

    def forward(self, *args, **kwargs) -> tuple[Any, ...]:
        for m in self:
            args = m(*args, **kwargs)
        return args
