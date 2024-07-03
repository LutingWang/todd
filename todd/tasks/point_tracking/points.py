__all__ = [
    'Point',
    'Points',
    'FlattenPoints',
]

from dataclasses import astuple, dataclass

import torch

from ..utils import FlattenMixin, NormalizeMixin, TensorWrapper


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def inside(self, image_wh: tuple[int, int]) -> bool:
        return (0, 0) <= astuple(self) <= image_wh


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

    def inside(self) -> torch.Tensor:
        tensor = self.normalize().to_tensor()
        return 0 <= tensor.min(-1) & tensor.max(-1) <= 1


class FlattenPoints(FlattenMixin[Point], Points):
    pass
