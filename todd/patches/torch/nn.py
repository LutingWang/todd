__all__ = [
    'ModuleList',
    'ModuleDict',
]

from torch import nn


class ModuleList(nn.ModuleList):

    def forward(self, *args, **kwargs) -> list[nn.Module]:
        return [m(*args, **kwargs) for m in self]


class ModuleDict(nn.ModuleDict):

    def forward(self, *args, **kwargs) -> dict[str, nn.Module]:
        return {k: m(*args, **kwargs) for k, m in self.items()}
