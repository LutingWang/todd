__all__ = [
    'Module',
    'ModuleList',
    'ModuleDict',
    'Sequential',
    'get_rank',
    'get_local_rank',
    'get_world_size',
]

import importlib.util
import os
from typing import TYPE_CHECKING

import torch.distributed as dist

# TODO: remove mmcv dependency
if importlib.util.find_spec('mmcv') and not TYPE_CHECKING:
    try:
        from mmcv.runner import BaseModule as Module
        from mmcv.runner import ModuleDict, ModuleList, Sequential
    except Exception:
        from torch.nn import Module, ModuleDict, ModuleList, Sequential
else:
    from torch.nn import Module, ModuleDict, ModuleList, Sequential


def get_rank(*args, **kwargs) -> int:
    if dist.is_initialized():
        return dist.get_rank(*args, **kwargs)
    return 0


def get_local_rank(*args, **kwargs) -> int:
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return get_rank(*args, **kwargs)


def get_world_size(*args, **kwargs) -> int:
    if dist.is_initialized():
        return dist.get_world_size(*args, **kwargs)
    return 1
