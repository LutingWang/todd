import itertools
from enum import Enum, auto
from functools import partial
from typing import Any, Iterable, Iterator, Optional, TypeVar

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..base import getattr_recur

DEFAULT_STOCHASTIC_MODULE_TYPES = [_BatchNorm, nn.Dropout]

try:
    from timm.models.layers import DropPath

    DEFAULT_STOCHASTIC_MODULE_TYPES.append(DropPath)
except Exception:
    pass


class NoGradMode(Enum):
    ALL = auto()
    PARTIAL = auto()
    NONE = auto()

    def no_grad(self, model: nn.Module, **kwargs) -> nn.Module:
        if self == NoGradMode.ALL:
            return model.requires_grad_(False)
        if self == NoGradMode.NONE:
            return model

        modules: Iterator[nn.Module]
        if self == NoGradMode.PARTIAL:
            modules = map(
                partial(getattr_recur, model),
                kwargs['module_names'],
            )
        else:
            raise NotImplementedError(f"{self} is not implemented")

        for module in modules:
            module.requires_grad_(False)
        return model


class EvalMode(Enum):
    ALL = auto()
    DETERMINISTIC_AND_PARTIAL = auto()
    DETERMINISTIC = auto()
    PARTIAL = auto()
    PARTIALLY_DETERMINISTIC = auto()
    NONE = auto()

    def eval(
        self,
        model: nn.Module,
        **kwargs,
    ) -> nn.Module:
        if self == EvalMode.ALL:
            return model.eval()
        if self == EvalMode.NONE:
            return model

        stochastic_module_types: Iterable[Any] = kwargs.get(
            'stochastic_module_types',
            DEFAULT_STOCHASTIC_MODULE_TYPES,
        )

        def stochastic(module: nn.Module) -> bool:
            return any(
                isinstance(module, smt) for smt in stochastic_module_types
            )

        modules: Iterator[nn.Module]
        if self == EvalMode.DETERMINISTIC:
            modules = filter(stochastic, model.modules())
        elif self == EvalMode.PARTIAL:
            modules = map(
                partial(getattr_recur, model),
                kwargs['module_names'],
            )
        elif self == EvalMode.PARTIALLY_DETERMINISTIC:
            modules = map(
                partial(getattr_recur, model),
                kwargs['module_names'],
            )
            modules = filter(stochastic, modules)
        elif self == EvalMode.DETERMINISTIC_AND_PARTIAL:
            named_modules = map(
                partial(getattr_recur, model),
                kwargs['module_names'],
            )
            stochastic_modules = filter(stochastic, model.modules())
            modules = itertools.chain(named_modules, stochastic_modules)
        else:
            raise NotImplementedError(f'EvalMode {self} not implemented')

        for module in modules:
            module.eval()
        return model


def freeze_model(model: nn.Module, no_grad=..., eval_=...) -> nn.Module:
    if no_grad is ...:
        no_grad = dict(mode='ALL')
    if no_grad is not None:
        no_grad = dict(no_grad)
        no_grad_mode_name: str = no_grad.pop('mode')
        no_grad_mode = NoGradMode[no_grad_mode_name.upper()]
        no_grad_mode.no_grad(model, **no_grad)
    if eval_ is ...:
        eval_ = dict(mode='ALL')
    if eval_ is not None:
        eval_ = dict(eval_)
        eval_mode_name: str = eval_.pop('mode')
        eval_mode = EvalMode[eval_mode_name.upper()]
        eval_mode.eval(model, **eval_)
    return model


T = TypeVar('T', bound='FrozenMixin')


class FrozenMixin(nn.Module):

    def __init__(
        self,
        *args,
        freeze_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._freeze_cfg = freeze_cfg or dict()
        freeze_model(self, **self._freeze_cfg)

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        result = super().requires_grad_(requires_grad)
        freeze_model(self, **self._freeze_cfg)
        return result

    def train(self: T, mode: bool = True) -> T:
        result = super().train(mode)
        freeze_model(self, **self._freeze_cfg)
        return result
