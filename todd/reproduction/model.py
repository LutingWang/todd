__all__ = [
    'get_modules',
    'no_grad',
    'eval_',
    'freeze',
    'FrozenMixin',
]

from typing import Any, Sequence, TypeVar

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..base import getattr_recur

DEFAULT_STOCHASTIC_MODULE_TYPES: tuple[Any, ...] = (_BatchNorm, nn.Dropout)

try:
    from timm.models.layers import DropPath

    DEFAULT_STOCHASTIC_MODULE_TYPES += DropPath,
except Exception:
    pass


def get_modules(
    model: nn.Module,
    names: Sequence[str] = tuple(),
    types: Sequence[type] = tuple(),
) -> list[nn.Module]:
    if not names and not types:
        return [model]
    if not types:
        return [getattr_recur(model, name) for name in names]
    types = tuple(types)
    if not names:
        return [
            module for module in model.modules() if isinstance(module, types)
        ]
    modules = [getattr_recur(model, name) for name in names]
    modules = [module for module in modules if isinstance(module, types)]
    return modules


def no_grad(model: nn.Module, **kwargs) -> None:
    for module in get_modules(model, **kwargs):
        module.requires_grad_(False)


def eval_(model: nn.Module, **kwargs) -> None:
    for module in get_modules(model, **kwargs):
        module.eval()


def freeze(model: nn.Module, **kwargs) -> None:
    no_grad(model, **kwargs)
    eval_(model, **kwargs)


T = TypeVar('T', bound='FrozenMixin')


class FrozenMixin(nn.Module):

    def __init__(
        self,
        no_grad_config: dict | None = None,
        eval_config: dict | None = None,
    ) -> None:
        self._no_grad = no_grad_config
        self._eval = eval_config
        if no_grad_config:
            no_grad(self, **no_grad_config)
        if eval_config:
            eval_(self, **eval_config)

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        result = super().requires_grad_(requires_grad)
        if self._no_grad:
            no_grad(self, **self._no_grad)
        return result

    def train(self: T, mode: bool = True) -> T:
        result = super().train(mode)
        if self._eval:
            eval_(self, **self._eval)
        return result
