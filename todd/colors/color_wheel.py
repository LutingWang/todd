__all__ = [
    'color_wheel',
]

from functools import cache

import torch


def zeros(i: int) -> torch.Tensor:
    return torch.zeros(i, dtype=torch.uint8)


def full(i: int) -> torch.Tensor:
    return torch.full((i, ), 255, dtype=torch.uint8)


def arange(i: int) -> torch.Tensor:
    tensor = torch.arange(i) / i * 255
    return tensor.type(torch.uint8)


@cache
def color_wheel(
    ry: int = 15,
    yg: int = 6,
    gc: int = 4,
    cb: int = 11,
    bm: int = 13,
    mr: int = 6,
) -> torch.tensor:
    return torch.cat([
        torch.stack([full(ry), arange(ry), zeros(ry)], dim=-1),
        torch.stack(
            [255 - arange(yg), full(yg), zeros(yg)],
            dim=-1,
        ),
        torch.stack([zeros(gc), full(gc), arange(gc)], dim=-1),
        torch.stack(
            [zeros(cb), 255 - arange(cb),
             full(cb)],
            dim=-1,
        ),
        torch.stack([arange(bm), zeros(bm), full(bm)], dim=-1),
        torch.stack(
            [full(mr), zeros(mr), 255 - arange(mr)],
            dim=-1,
        ),
    ])
