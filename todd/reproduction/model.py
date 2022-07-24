import itertools
from enum import Enum, auto
from typing import Optional

import torch.nn as nn
from timm.models.layers import DropPath
from torch.nn.modules.batchnorm import _BatchNorm


class NoGradMode(Enum):
    ALL = auto()
    PARTIAL = auto()
    NONE = auto()

    def no_grad(self, model: nn.Module, **kwargs) -> nn.Module:
        if self == NoGradMode.ALL:
            return model.requires_grad_(False)
        elif self == NoGradMode.PARTIAL:
            for module_name in kwargs['modules']:
                module: nn.Module = getattr(model, module_name)
                module.requires_grad_(False)
            return model
        elif self == NoGradMode.NONE:
            return model
        raise NotImplementedError(f"{self} is not implemented")


DEFAULT_STOCHASTIC_MODULE_TYPES = [_BatchNorm, nn.Dropout, DropPath]


class EvalMode(Enum):
    ALL = auto()
    DETERMINISTIC_AND_PARTIAL = auto()
    DETERMINISTIC = auto()
    PARTIAL = auto()
    PARTIALLY_DETERMINISTIC = auto()
    NONE = auto()

    def eval(self, model: nn.Module, **kwargs) -> nn.Module:
        if self == EvalMode.ALL:
            return model.eval()
        elif self == EvalMode.NONE:
            return model
        # TODO: refactor
        modules = (
            model.modules() if self == EvalMode.DETERMINISTIC else
            (getattr(model, module) for module in kwargs['modules'])
        )
        if self != EvalMode.PARTIAL:
            sms = kwargs.get(
                'stochastic_module_types',
                DEFAULT_STOCHASTIC_MODULE_TYPES,
            )
            if self in [
                EvalMode.DETERMINISTIC,
                EvalMode.PARTIALLY_DETERMINISTIC,
            ]:
                modules = (
                    module for module in modules
                    if any(isinstance(module, sm) for sm in sms)
                )
            elif self == EvalMode.DETERMINISTIC_AND_PARTIAL:
                modules = itertools.chain(
                    modules,
                    (
                        module for module in model.modules()
                        if any(isinstance(module, sm) for sm in sms)
                    ),
                )
            else:
                raise NotImplementedError(f'EvalMode {self} not implemented')
        for module in modules:
            module.eval()
        return model


def freeze_model(
    model: nn.Module,
    no_grad: Optional[dict] = ...,  # type: ignore[assignment]
    eval_: Optional[dict] = ...,  # type: ignore[assignment]
) -> nn.Module:
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


class FrozenMixin(nn.Module):

    def __init__(
        self,
        *args,
        freeze_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if freeze_cfg is None:
            return
        freeze_model(self, **freeze_cfg)
        self._eval_cfg = freeze_cfg.get('eval_')

    def train(self, mode: bool = True):
        result = super().train(mode)
        if hasattr(self, '_eval_cfg'):
            freeze_model(self, no_grad=None, eval_=self._eval_cfg)
        return result
