import torch.distributed as dist

__all__ = [
    'get_rank',
    'get_world_size',
]


def get_rank(*args, **kwargs) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(*args, **kwargs)
    return 0


def get_world_size(*args, **kwargs) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(*args, **kwargs)
    return 1
