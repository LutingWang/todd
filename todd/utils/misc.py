from contextlib import AbstractContextManager
from typing import Any, Optional

from mmcv.runner import BaseModule
import torch.nn as nn


def getattr_recur(obj: Any, attr: str, allow_list: bool = False) -> Any:
    if attr == '':
        return obj
    if not allow_list:
        return eval('obj.' + attr)
    for a in attr.split('.'):
        obj = obj[int(a)] if a.isnumeric() else eval('obj.' + a)
    return obj


def setattr_recur(obj: Any, attr: str, value: Any, allow_list: bool = False):
    if '.' in attr:
        attr, tmp = attr.rsplit('.', 1)
        obj = getattr_recur(obj, attr, allow_list)
        attr = tmp
    if allow_list and attr.isnumeric:
        obj[int(attr)] = value
    else:
        setattr(obj, attr, value)


class setattr_temp(AbstractContextManager):
    def __init__(self, obj: Any, attr: str, value: Any, allow_list: bool = False):
        self._obj = obj
        self._attr = attr
        self._value = value
        self._allow_list = allow_list

    def __enter__(self):
        self._prev = getattr_recur(self._obj, self._attr, self._allow_list)
        setattr_recur(self._obj, self._attr, self._value, self._allow_list)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        setattr_recur(self._obj, self._attr, self._prev, self._allow_list)


def freeze_model(model: nn.Module):
    model.eval()
    model.requires_grad_(False)


def build(cls, cfg, **kwargs) -> Optional['BaseModule']:
    if cfg is None: return None
    module = cfg if isinstance(cfg, cls) else cls(cfg, **kwargs)
    return module


BaseModule.build = classmethod(build)
