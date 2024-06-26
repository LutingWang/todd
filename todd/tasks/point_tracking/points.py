__all__ = [
    'Point',
    'Points',
    'FlattenPoints',
]

from dataclasses import dataclass

import torch

from ..utils import FlattenMixin, NormalizeMixin, TensorWrapper


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def inside(self, image_wh: tuple[int, int]) -> bool:
        w, h = image_wh
        return 0 <= self.x < w and 0 <= self.y < h


class Points(NormalizeMixin[Point], TensorWrapper[Point]):
    OBJECT_DIMENSIONS = 1

    @classmethod
    def to_object(cls, tensor: torch.Tensor) -> Point:
        x, y = tensor.int().tolist()
        return Point(x, y)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self._tensor.shape[-1] == 2

    def _scale(self, ratio_xy: tuple[float, ...], /) -> torch.Tensor:
        return self._tensor * self._tensor.new_tensor(
            ratio_xy,
            dtype=torch.float32,
        )

    def flatten(self) -> 'FlattenPoints':
        args, kwargs = self.copy(self._flatten()).__getstate__()
        return FlattenPoints(*args, **kwargs)


class FlattenPoints(FlattenMixin[Point], Points):
    pass
