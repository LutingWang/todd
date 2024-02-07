__all__ = [
    'get_rank',
    'get_local_rank',
    'get_world_size',
    'all_gather',
    'all_gather_',
    'Shape',
    'ModuleList',
    'ModuleDict',
]

import functools
import itertools
import operator
import os
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


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


class Shape:

    @classmethod
    def module(cls, module: nn.Module, x: torch.Tensor) -> tuple[int, ...]:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return cls.conv(module, x)
        raise TypeError(f"Unknown type {type(module)}")

    @staticmethod
    def _conv(
        x: int,
        padding: int,
        dilation: int,
        kernel_size: int,
        stride: int,
    ) -> int:
        x += 2 * padding - dilation * (kernel_size - 1) - 1
        return x // stride + 1

    @classmethod
    def conv(
        cls,
        module: nn.Conv1d | nn.Conv2d | nn.Conv3d,
        x: torch.Tensor,
    ) -> tuple[int, ...]:
        b, c, *shape = x.shape

        assert c == module.in_channels
        c = module.out_channels

        return b, c, *itertools.starmap(
            cls._conv,
            zip(
                shape,
                module.padding,
                module.dilation,
                module.kernel_size,
                module.stride,
            ),
        )


class ModuleList(nn.ModuleList):

    def forward(self, *args, **kwargs) -> list:
        return [m(*args, **kwargs) for m in self]


class ModuleDict(nn.ModuleDict):

    def forward(self, *args, **kwargs) -> dict[str, Any]:
        return {k: m(*args, **kwargs) for k, m in self.items()}
