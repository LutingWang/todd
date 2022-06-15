import torch
import torch.distributed as dist

from ..logger import get_logger

__all__ = [
    'get_rank',
    'get_world_size',
]

if torch.__version__ < '1.7.0':
    get_logger().warning(
        "Monkey patching `torch.maximum` and `torch.minimum`.",
    )
    torch.maximum = torch.max
    torch.Tensor.maximum = torch.Tensor.max
    torch.minimum = torch.min
    torch.Tensor.minimum = torch.Tensor.min


def get_rank(*args, **kwargs) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(*args, **kwargs)
    return 0


def get_world_size(*args, **kwargs) -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(*args, **kwargs)
    return 1
