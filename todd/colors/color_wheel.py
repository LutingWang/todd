__all__ = [
    'ColorWheel',
]

import torch


def zeros(i: int) -> torch.Tensor:
    return torch.zeros(i, dtype=torch.uint8)


def full(i: int) -> torch.Tensor:
    return torch.full((i, ), 255, dtype=torch.uint8)


def arange(i: int) -> torch.Tensor:
    tensor = torch.arange(i) / i * 255
    return tensor.type(torch.uint8)


class ColorWheel:

    def __init__(
        self,
        ry: int = 15,
        yg: int = 6,
        gc: int = 4,
        cb: int = 11,
        bm: int = 13,
        mr: int = 6,
    ) -> None:
        self._color_wheel = torch.cat([
            torch.stack([full(ry), arange(ry), zeros(ry)], -1),
            torch.stack(
                [255 - arange(yg), full(yg),
                 zeros(yg)],
                -1,
            ),
            torch.stack([zeros(gc), full(gc), arange(gc)], -1),
            torch.stack(
                [zeros(cb), 255 - arange(cb),
                 full(cb)],
                -1,
            ),
            torch.stack(
                [arange(bm), zeros(bm), full(bm)],
                -1,
            ),
            torch.stack([full(mr), zeros(mr), 255 - arange(mr)], -1),
        ])

    def __len__(self) -> int:
        return self._color_wheel.shape[0]

    def __getitem__(self, index: torch.Tensor) -> torch.Tensor:
        index = index % len(self)
        floor = index.floor().int()
        ceil = (floor + 1) % len(self)
        alpha = index - floor
        return ((1 - alpha).unsqueeze(-1) * self._color_wheel[floor]
                + alpha.unsqueeze(-1) * self._color_wheel[ceil])
