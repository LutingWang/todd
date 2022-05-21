from enum import Enum, auto
import itertools
from typing import List, Optional, cast

from timm.models.layers import DropPath
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn as nn

from ..utils import getattr_recur


class NoGradMode(Enum):
    ALL = auto()
    PARTIAL = auto()
    NONE = auto()

    def no_grad(self, model: nn.Module, **kwargs) -> nn.Module:
        modules: List[nn.Module] = []
        if self == NoGradMode.ALL:
            modules.append(model)
        elif self == NoGradMode.PARTIAL:
            for module in kwargs['modules']:
                module = getattr_recur(model, module)
                modules.append(module)
        elif self == NoGradMode.NONE:
            pass
        for module in modules:
            module.requires_grad_(False)
        return model


DEFAULT_STOCHASTIC_MODULE_TYPES = [_BatchNorm, nn.Dropout, DropPath]


class EvalMode(Enum):
    ALL = auto()
    DETERMINISTIC_AND_PARTIAL = auto()
    DETERMINISTIC = auto()
    PARTIAL = auto()
    PARTIALLY_DETERMINISTIC = auto()
    NONE = auto()

    def eval(self, model: nn.Module, **kwargs) -> nn.Module:
        modules: List[nn.Module] = []
        if self == EvalMode.ALL:
            modules.append(model)
        elif self == EvalMode.NONE:
            pass
        else:
            modules = (
                model.modules() if self == EvalMode.DETERMINISTIC else 
                (getattr_recur(model, module) for module in kwargs['modules'])
            )
            if self != EvalMode.PARTIAL:
                sms = kwargs.get('stochastic_module_types', DEFAULT_STOCHASTIC_MODULE_TYPES)
                if self in [EvalMode.DETERMINISTIC, EvalMode.PARTIALLY_DETERMINISTIC]:
                    modules = (module for module in modules if any(isinstance(module, sm) for sm in sms))
                elif self == EvalMode.DETERMINISTIC_AND_PARTIAL:
                    modules = itertools.chain(modules, (module for module in model.modules() if any(isinstance(module, sm) for sm in sms)))
                else:
                    raise NotImplementedError(f'EvalMode {self} not implemented')
        for module in modules:
            module.eval()
        return model


def freeze_model(
    model: nn.Module, 
    no_grad: Optional[dict] = None,
    eval_: Optional[dict] = None,
) -> nn.Module:
    if no_grad is not None:
        no_grad = dict(no_grad)
        no_grad_mode = cast(str, no_grad.pop('mode'))
        no_grad_mode = cast(NoGradMode, getattr(NoGradMode, no_grad_mode.upper()))
        no_grad_mode.no_grad(model, **no_grad)
    if eval_ is not None:
        eval_ = dict(eval_)
        eval_mode = cast(str, eval_.pop('mode'))
        eval_mode = cast(EvalMode, getattr(EvalMode, eval_mode.upper()))
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
        if freeze_cfg is not None:
            freeze_model(self, **freeze_cfg)
        self._eval_cfg = freeze_cfg.get('eval_')

    def train(self, mode: bool = True):
        result = super().train(mode)
        if self._eval_cfg is not None:
            freeze_model(self, eval_=self._eval_cfg)
        return result
    