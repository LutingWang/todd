__all__ = [
    'Module',
    'ModuleList',
    'ModuleDict',
    'Sequential',
    'get_rank',
    'get_local_rank',
    'get_world_size',
]

import os

import torch.distributed as dist

try:
    from mmcv.runner import BaseModule as Module
    from mmcv.runner import ModuleDict, ModuleList, Sequential
except Exception:
    from torch.nn import Module, ModuleDict, ModuleList, Sequential


def get_rank(*args, **kwargs) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(*args, **kwargs)
    return 0


def get_local_rank() -> int:
    return int(os.getenv('LOCAL_RANK', -1))


def get_world_size(*args, **kwargs) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(*args, **kwargs)
    return 1
