__all__ = [
    'ModuleList',
    'ModuleDict',
    'Sequential',
    'training_modules',
    'named_training_modules',
    'trainable_parameters',
    'named_trainable_parameters',
]

from typing import Any, Generator

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


def training_modules(
    module: nn.Module,
    *args,
    **kwargs,
) -> Generator[nn.Module, None, None]:
    for m in module.modules(*args, **kwargs):
        if m.training:
            yield m


def named_training_modules(
    module: nn.Module,
    *args,
    **kwargs,
) -> Generator[tuple[str, nn.Module], None, None]:
    for name, m in module.named_modules(*args, **kwargs):
        if m.training:
            yield name, m


def trainable_parameters(
    module: nn.Module,
    *args,
    **kwargs,
) -> Generator[nn.Parameter, None, None]:
    for parameter in module.parameters(*args, **kwargs):
        if parameter.requires_grad:
            yield parameter


def named_trainable_parameters(
    module: nn.Module,
    *args,
    **kwargs,
) -> Generator[tuple[str, nn.Parameter], None, None]:
    for name, parameter in module.named_parameters(*args, **kwargs):
        if parameter.requires_grad:
            yield name, parameter
