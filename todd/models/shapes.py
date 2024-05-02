__all__ = [
    'Shape',
]

import itertools

import torch
from torch import nn

# TODO: update


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
