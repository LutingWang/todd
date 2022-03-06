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


def freeze_model(model: nn.Module):
    model.eval()
    model.requires_grad_(False)


def build(cls, cfg, **kwargs) -> Optional['BaseModule']:
    if cfg is None: return None
    module = cfg if isinstance(cfg, cls) else cls(cfg, **kwargs)
    return module


BaseModule.build = classmethod(build)
