# pylint: disable=invalid-name

__all__ = [
    'BBox',
    'BBoxes',
    'BBoxesXY__',
    'BBoxesCXCY__',
    'BBoxes__XY',
    'BBoxes__WH',
    'BBoxesXYXY',
    'BBoxesXYWH',
    'BBoxesCXCYWH',
    'FlattenBBoxesMixin',
    'FlattenBBoxesXYXY',
    'FlattenBBoxesXYWH',
    'FlattenBBoxesCXCYWH',
]

from abc import ABC, abstractmethod
from typing import TypeVar
from typing_extensions import Self

import einops.layers.torch
import torch

from ..utils import FlattenMixin, NormalizeMixin, TensorWrapper
from .registries import ODBBoxesRegistry

BBox = tuple[float, float, float, float]
T = TypeVar('T', bound='BBoxes')


class BBoxes(NormalizeMixin[BBox], TensorWrapper[BBox], ABC):
    OBJECT_DIMENSIONS = 1

    @classmethod
    def to_object(cls, tensor: torch.Tensor) -> BBox:
        return tuple(tensor.tolist())

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self._tensor.shape[-1] == 4

    def _scale(self, ratio_xy: tuple[float, ...], /) -> torch.Tensor:
        return self._tensor * self._tensor.new_tensor(ratio_xy * 2)

    @property
    @abstractmethod
    def left(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def right(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def top(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def bottom(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def width(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def height(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def center_x(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def center_y(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def lt(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def rb(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def wh(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def center(self) -> torch.Tensor:
        pass

    @property
    def area(self) -> torch.Tensor:
        return self.width * self.height

    @classmethod
    @abstractmethod
    def _from1(cls, bboxes: 'BBoxes') -> torch.Tensor:
        """Convert to the first two coordinates of the bboxes.

        Args:
            bboxes: the bboxes to convert.

        Returns:
            A tensor representing the first two coordinates of the bboxes.
        """

    @classmethod
    @abstractmethod
    def _from2(cls, bboxes: 'BBoxes') -> torch.Tensor:
        """Convert to the last two coordinates of the bboxes.

        Args:
            bboxes: the bboxes to convert.

        Returns:
            A tensor representing the last two coordinates of the bboxes.
        """

    @classmethod
    def from_(cls, bboxes: 'BBoxes') -> Self:
        from1 = cls._from1(bboxes)
        from2 = cls._from2(bboxes)
        from_ = torch.cat([from1, from2], -1)
        (_, *args), kwargs = bboxes.__getstate__()
        return cls(from_, *args, **kwargs)

    def to(self, cls: type[T]) -> T:
        return cls.from_(self)

    def translate(self, offset_xy: tuple[float, float] | torch.Tensor) -> Self:
        if isinstance(offset_xy, tuple):
            offset_xy = self._tensor.new_tensor(offset_xy)
        bboxes = self.to(BBoxesXYXY)
        tensor = bboxes._tensor + torch.cat([offset_xy, offset_xy], -1)
        bboxes = bboxes.copy(tensor)
        return bboxes.to(self.__class__)

    def round(self) -> Self:
        bboxes = self.to(BBoxesXYXY)
        if normalized := bboxes._normalized:
            bboxes = bboxes.denormalize()
        lt = bboxes.lt.floor()
        rb = bboxes.rb.ceil()
        tensor = torch.cat([lt, rb], -1)
        bboxes = bboxes.copy(tensor)
        if normalized:
            bboxes = bboxes.normalize()
        return bboxes.to(self.__class__)

    def expand(self, ratio_wh: tuple[float, float] | torch.Tensor) -> Self:
        if isinstance(ratio_wh, tuple):
            ratio_wh = self._tensor.new_tensor(ratio_wh)
        bboxes = self.to(BBoxesCXCYWH)
        tensor = torch.cat([bboxes.center, bboxes.wh * ratio_wh], -1)
        bboxes = bboxes.copy(tensor)
        return bboxes.to(self.__class__)

    def clamp(self) -> Self:
        bboxes = self.to(BBoxesXYXY)
        if bboxes._normalized:
            tensor = bboxes._tensor.clamp(0, 1)
        else:
            tensor = bboxes._tensor.clamp_min(0)
            tensor = tensor.clamp_max(
                bboxes._tensor.new_tensor(bboxes.divisor * 2),
            )
        bboxes = bboxes.copy(tensor)
        return bboxes.to(self.__class__)

    def indices(
        self,
        *,
        min_area: float | None = None,
        min_wh: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        indices = self._tensor.new_ones(self.shape, dtype=torch.bool)
        if min_area is not None:
            indices &= self.area >= min_area
        if min_wh is not None:
            indices &= (self.wh >= torch.tensor(min_wh)).all(-1)
        return indices

    def pairwise_intersections(self, other: 'BBoxes') -> torch.Tensor:
        lt = torch.maximum(self.lt, other.lt)
        rb = torch.minimum(self.rb, other.rb)
        wh = rb - lt
        wh = wh.clamp_min_(0)
        return wh[:, 0] * wh[:, 1]

    def _pairwise_unions(
        self,
        other: 'BBoxes',
        intersections: torch.Tensor,
    ) -> torch.Tensor:
        return self.area + other.area - intersections

    def pairwise_unions(self, other: 'BBoxes') -> torch.Tensor:
        intersections = self.pairwise_intersections(other)
        return self._pairwise_unions(other, intersections)

    def pairwise_ious(
        self,
        other: 'BBoxes',
        eps: float = 1e-6,
    ) -> torch.Tensor:
        intersections = self.pairwise_intersections(other)
        unions = self._pairwise_unions(other, intersections)
        unions = unions.clamp_min(eps)
        return intersections / unions

    def to_mask(self) -> torch.Tensor:
        w, h = self.divisor
        x = torch.arange(w, device=self._tensor.device)
        y = torch.arange(h, device=self._tensor.device)
        rearrange = einops.layers.torch.Rearrange('... -> ... 1')
        x_mask = (rearrange(self.left) <= x) & (x <= rearrange(self.right))
        y_mask = (rearrange(self.top) <= y) & (y <= rearrange(self.bottom))

        x_mask = einops.rearrange(x_mask, '... d -> ... 1 d')
        y_mask = rearrange(y_mask)
        mask = x_mask & y_mask
        return mask


class BBoxesXY__(BBoxes, ABC):

    @property
    def left(self) -> torch.Tensor:
        return self._tensor[:, 0]

    @property
    def top(self) -> torch.Tensor:
        return self._tensor[:, 1]

    @property
    def lt(self) -> torch.Tensor:
        return self._tensor[:, :2]

    @classmethod
    def _from1(cls, bboxes: BBoxes) -> torch.Tensor:
        return bboxes.lt


class BBoxesCXCY__(BBoxes, ABC):

    @property
    def center_x(self) -> torch.Tensor:
        return self._tensor[:, 0]

    @property
    def center_y(self) -> torch.Tensor:
        return self._tensor[:, 1]

    @property
    def center(self) -> torch.Tensor:
        return self._tensor[:, :2]

    @classmethod
    def _from1(cls, bboxes: BBoxes) -> torch.Tensor:
        return bboxes.center


class BBoxes__XY(BBoxes, ABC):  # noqa: N801

    @property
    def right(self) -> torch.Tensor:
        return self._tensor[:, 2]

    @property
    def bottom(self) -> torch.Tensor:
        return self._tensor[:, 3]

    @property
    def rb(self) -> torch.Tensor:
        return self._tensor[:, 2:]

    @classmethod
    def _from2(cls, bboxes: BBoxes) -> torch.Tensor:
        return bboxes.rb


class BBoxes__WH(BBoxes, ABC):  # noqa: N801

    @property
    def width(self) -> torch.Tensor:
        return self._tensor[:, 2]

    @property
    def height(self) -> torch.Tensor:
        return self._tensor[:, 3]

    @property
    def wh(self) -> torch.Tensor:
        return self._tensor[:, 2:]

    @classmethod
    def _from2(cls, bboxes: BBoxes) -> torch.Tensor:
        return bboxes.wh


@ODBBoxesRegistry.register_()
class BBoxesXYXY(BBoxesXY__, BBoxes__XY):

    @property
    def width(self) -> torch.Tensor:
        return self.right - self.left

    @property
    def height(self) -> torch.Tensor:
        return self.bottom - self.top

    @property
    def center_x(self) -> torch.Tensor:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> torch.Tensor:
        return (self.top + self.bottom) / 2

    @property
    def wh(self) -> torch.Tensor:
        return self.rb - self.lt

    @property
    def center(self) -> torch.Tensor:
        return (self.lt + self.rb) / 2

    def flatten(self) -> 'FlattenBBoxesXYXY':
        args, kwargs = self.copy(self._flatten()).__getstate__()
        return FlattenBBoxesXYXY(*args, **kwargs)


@ODBBoxesRegistry.register_()
class BBoxesXYWH(BBoxesXY__, BBoxes__WH):

    @property
    def right(self) -> torch.Tensor:
        return self.left + self.width

    @property
    def bottom(self) -> torch.Tensor:
        return self.top + self.height

    @property
    def center_x(self) -> torch.Tensor:
        return self.left + self.width / 2

    @property
    def center_y(self) -> torch.Tensor:
        return self.top + self.height / 2

    @property
    def rb(self) -> torch.Tensor:
        return self.lt + self.wh

    @property
    def center(self) -> torch.Tensor:
        return self.lt + self.wh / 2

    def flatten(self) -> 'FlattenBBoxesXYWH':
        args, kwargs = self.copy(self._flatten()).__getstate__()
        return FlattenBBoxesXYWH(*args, **kwargs)


@ODBBoxesRegistry.register_()
class BBoxesCXCYWH(BBoxesCXCY__, BBoxes__WH):

    @property
    def left(self) -> torch.Tensor:
        return self.center_x - self.width / 2

    @property
    def right(self) -> torch.Tensor:
        return self.center_x + self.width / 2

    @property
    def top(self) -> torch.Tensor:
        return self.center_y - self.height / 2

    @property
    def bottom(self) -> torch.Tensor:
        return self.center_y + self.height / 2

    @property
    def lt(self) -> torch.Tensor:
        return self.center - self.wh / 2

    @property
    def rb(self) -> torch.Tensor:
        return self.center + self.wh / 2

    def flatten(self) -> 'FlattenBBoxesCXCYWH':
        args, kwargs = self.copy(self._flatten()).__getstate__()
        return FlattenBBoxesCXCYWH(*args, **kwargs)


class FlattenBBoxesMixin(FlattenMixin[BBox], BBoxes, ABC):

    def intersections(self, other: 'FlattenBBoxesMixin') -> torch.Tensor:
        r"""Intersections.

        Args:
            other: :math:`n' \times 4`.

        Returns:
            :math:`n \times n'`.
        """
        lt = torch.maximum(  # [n, n', 2]
            einops.rearrange(self.lt, 'n1 lt -> n1 1 lt'),
            einops.rearrange(other.lt, 'n2 lt -> 1 n2 lt'),
        )
        rb = torch.minimum(  # [n, n', 2]
            einops.rearrange(self.rb, 'n1 rb -> n1 1 rb'),
            einops.rearrange(other.rb, 'n2 rb -> 1 n2 rb'),
        )
        wh = rb - lt
        wh = wh.clamp_min_(0)
        return wh[..., 0] * wh[..., 1]

    def __and__(self, other: 'FlattenBBoxesMixin') -> torch.Tensor:
        return self.intersections(other)

    def _unions(
        self,
        other: 'FlattenBBoxesMixin',
        intersections: torch.Tensor,
    ) -> torch.Tensor:
        r"""Unions.

        Args:
            other: :math:`n' \times 4`.
            intersections: :math:`n \times n'`

        Returns:
            :math:`n \times n'`.
        """
        return self.area[:, None] + other.area[None, :] - intersections

    def unions(self, other: 'FlattenBBoxesMixin') -> torch.Tensor:
        r"""Unions.

        Args:
            other: :math:`n' \times 4`.

        Returns:
            :math:`n \times n'`.
        """
        intersections = self.intersections(other)
        return self._unions(other, intersections)

    def __or__(self, other: 'FlattenBBoxesMixin') -> torch.Tensor:
        return self.unions(other)

    def ious(
        self,
        other: 'FlattenBBoxesMixin',
        eps: float = 1e-6,
    ) -> torch.Tensor:
        r"""Intersections over unions.

        Args:
            other: :math:`n' \times 4`.
            eps: avoid division by zero.

        Returns:
            :math:`n \times n'`.
        """
        intersections = self.intersections(other)
        unions = self._unions(other, intersections)
        unions = unions.clamp_min(eps)
        return intersections / unions


@ODBBoxesRegistry.register_()
class FlattenBBoxesXYXY(FlattenBBoxesMixin, BBoxesXYXY):
    pass


@ODBBoxesRegistry.register_()
class FlattenBBoxesXYWH(FlattenBBoxesMixin, BBoxesXYWH):
    pass


@ODBBoxesRegistry.register_()
class FlattenBBoxesCXCYWH(FlattenBBoxesMixin, BBoxesCXCYWH):
    pass
