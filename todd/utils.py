from typing import Any, Optional

from mmcv.runner import BaseModule
import torch.nn as nn


_iter = None


def init_iter(iter_: int = 0):
    global _iter
    assert _iter is None, f"iter={_iter} has been initialized."
    _iter = iter_


def get_iter() -> Optional[int]:
    return _iter


def inc_iter():
    global _iter
    _iter += 1


def getattr_recur(obj: Any, attr: str) -> Any:
    return eval('obj.' + attr)


def freeze_model(model: nn.Module):
    model.eval()
    model.requires_grad_(False)


def build(cls, cfg, **kwargs) -> Optional['BaseModule']:
    if cfg is None: return None
    module = cfg if isinstance(cfg, cls) else cls(cfg, **kwargs)
    return module


BaseModule.build = classmethod(build)