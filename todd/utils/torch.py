__all__ = [
    'get_rank',
    'get_local_rank',
    'get_world_size',
    'all_gather',
    'all_gather_',
    'all_sync',
    'all_close',
    'set_epoch',
    'transfer_weight',
    'transfer_weights',
]

import functools
import operator
import os
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader


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


def all_gather_decorator(func):

    @functools.wraps(func)
    def wrapper(tensor: torch.Tensor, *args, **kwargs) -> list[torch.Tensor]:
        tensor = tensor.detach().clone()
        world_size = get_world_size()
        if world_size <= 1:
            return [tensor]
        return func(tensor, world_size, *args, **kwargs)

    return wrapper


@all_gather_decorator
def all_gather(
    tensor: torch.Tensor,
    world_size: int,
    *args,
    **kwargs,
) -> list[torch.Tensor]:
    tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor, *args, **kwargs)
    return tensors


@all_gather_decorator
def all_gather_(
    tensor: torch.Tensor,
    world_size: int,
    *args,
    **kwargs,
) -> list[torch.Tensor]:
    shape = tensor.shape
    shapes = [shape] * world_size
    dist.all_gather_object(shapes, shape)

    numel_list = [functools.reduce(operator.mul, s, 1) for s in shapes]
    max_numel = max(numel_list)

    container = tensor.new_empty(max_numel)
    containers = [torch.empty_like(container) for _ in range(world_size)]

    container[:tensor.numel()] = tensor.view(-1)
    dist.all_gather(containers, container, *args, **kwargs)

    tensors = [
        container[:numel].view(shape)
        for shape, numel, container in zip(shapes, numel_list, containers)
    ]
    return tensors


def all_sync(x: torch.Tensor) -> bool:
    if get_world_size() <= 1:
        return True
    x_prime = x.clone()
    dist.all_reduce(x)
    x /= get_world_size()
    return torch.allclose(x, x_prime)


def all_close(x: Any, y: Any, *args, **kwargs) -> bool:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    return torch.allclose(x, y, *args, **kwargs)


def set_epoch(dataloader: DataLoader, epoch: int) -> None:
    samplers = [
        dataloader.sampler,
        dataloader.batch_sampler,
        getattr(dataloader.batch_sampler, 'sampler', None),
    ]
    for sampler in samplers:
        set_epoch_ = getattr(sampler, 'set_epoch', None)
        if set_epoch_ is not None:
            set_epoch_(epoch)


def load(f, *args, directory=None, **kwargs):
    if directory is not None:
        f = os.path.join(directory, f)
    return torch.load(f, *args, **kwargs)


def transfer_weight(target: nn.Module, source: nn.Module) -> None:
    state_dict = source.state_dict()
    incompatible_keys = target.load_state_dict(state_dict, strict=False)
    if get_rank() == 0:
        from ..logger import logger  # pylint: disable=import-outside-toplevel
        logger.info(incompatible_keys)


def transfer_weights(models, weight_prefixes: Mapping[str, str]) -> None:
    from ..patches import get_  # pylint: disable=import-outside-toplevel
    for target_prefix, source_prefix in weight_prefixes.items():
        target = get_(models, target_prefix)
        source = get_(models, source_prefix)
        transfer_weight(target, source)
