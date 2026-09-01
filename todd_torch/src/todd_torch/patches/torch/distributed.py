__all__ = [
    'get_rank',
    'get_local_rank',
    'get_world_size',
    'all_gather',
    'all_gather_object',
]

import functools
import operator
import os

import torch
import torch.distributed as dist


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
def all_gather_object(
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
