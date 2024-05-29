# pylint: disable=invalid-name,self-cls-assignment

__all__ = [
    'BBoxes',
    'BBoxesXY__',
    'BBoxesCXCY__',
    'BBoxes__XY',
    'BBoxes__WH',
    'BBoxesXYXY',
    'BBoxesXYWH',
    'BBoxesCXCYWH',
]

import functools
from abc import ABC, abstractmethod
from typing import Any, Generator, TypeVar
from typing_extensions import Self

import einops
import torch

BBox = tuple[float, float, float, float]
T = TypeVar('T', bound='BBoxes')


class BBoxes(ABC):

    def __init__(
        self,
        bboxes: torch.Tensor,
        *,
        normalized: bool = False,
        image_wh: tuple[int, int] | None = None,
    ) -> None:
        r"""Initialize.

        Args:
            bboxes: :math:`n \times 4`.
            normalized: whether the bboxes are normalized.
            image_wh: the size of the image.

        Bounding boxes must be 2 dimensional and the second dimension must be
        of size 4.
        """
        if bboxes.ndim != 2:
            raise ValueError('bboxes must be 2-dim')
        if bboxes.shape[-1] != 4:
            raise ValueError('bboxes must have 4 columns')
        self._bboxes = bboxes
        self._normalized = normalized
        if image_wh is not None:
            self.set_image_wh(image_wh)

    def __len__(self) -> int:
        return self._bboxes.shape[0]

    def __iter__(self) -> Generator[BBox, None, None]:
        """Iterate over bboxes.

        Yields:
            One bbox.
        """
        yield from map(tuple, self._bboxes.tolist())

    def __repr__(self) -> str:
        kwargs = ''
        if self._normalized:
            kwargs += ', normalized=True'
        if self.has_image_wh:
            kwargs += f', image_wh={self._image_wh}'
        return f'{type(self).__name__}({self._bboxes}{kwargs})'

    def _copy(self, bboxes: torch.Tensor, **kwargs) -> Self:
        if self._normalized:
            kwargs.setdefault('normalized', True)
        if self.has_image_wh:
            kwargs.setdefault('image_wh', self._image_wh)
        return self.__class__(bboxes, **kwargs)

    def __getitem__(self, indices) -> Self:
        """Get specific bboxes.

        Args:
            indices: a index or multiple indices.

        Returns:
            If `indices` refers to a single box, return a ``tuple``.
            Otherwise, return ``BBoxes``.
        """
        bboxes = self._bboxes[indices]
        if bboxes.ndim == 1:
            bboxes = bboxes.unsqueeze(0)
        return self._copy(bboxes)

    def __add__(self, other: Self) -> Self:
        r"""Concatenate bboxes.

        Args:
            other: :math:`n' \times 4`.

        Returns:
            :math:`(n + n') \times 4`, where `n` is the length of `self`.
        """
        assert self._normalized == other._normalized
        if self.has_image_wh:
            assert self._image_wh == other._image_wh
        else:
            assert not other.has_image_wh
        bboxes = torch.cat([self._bboxes, other._bboxes])
        return self._copy(bboxes)

    @property
    def normalized(self) -> bool:
        return self._normalized

    @property
    def has_image_wh(self) -> bool:
        return hasattr(self, '_image_wh')

    @property
    def image_wh(self) -> tuple[int, int]:
        return self._image_wh

    def set_image_wh(
        self,
        image_wh: tuple[int, int],
        override: bool = False,
    ) -> None:
        if not self.has_image_wh or override:
            self._image_wh = image_wh
            return
        assert self._image_wh == image_wh, f"{self._image_wh} != {image_wh}"

    def to_tensor(self) -> torch.Tensor:
        return self._bboxes

    def _scale(self, ratio_xy: tuple[float, float]) -> torch.Tensor:
        return self._bboxes.new_tensor(ratio_xy * 2)

    def scale(self, ratio_xy: tuple[float, float]) -> Self:
        ratio = self._scale(ratio_xy)
        bboxes = self._bboxes * ratio
        return self._copy(bboxes)

    def __mul__(self, ratio_xy: tuple[float, float]) -> Self:
        return self.scale(ratio_xy)

    def __truediv__(self, ratio_xy: tuple[float, float]) -> Self:
        w, h = ratio_xy
        ratio_xy = (1 / w, 1 / h)
        return self.scale(ratio_xy)

    def _translate(self, offset_xy: tuple[float, float]) -> torch.Tensor:
        return self._bboxes.new_tensor(offset_xy * 2)

    def translate(self, offset_xy: tuple[float, float]) -> Self:
        offset = self._translate(offset_xy)
        bboxes = self._bboxes + offset
        return self._copy(bboxes)

    def normalize(self) -> Self:
        if self._normalized:
            return self._copy(self._bboxes)
        self = self / self.image_wh
        self._normalized = True
        return self

    def denormalize(self) -> Self:
        if not self._normalized:
            return self._copy(self._bboxes)
        self = self.scale(self.image_wh)
        self._normalized = False
        return self

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
        from_ = torch.cat([from1, from2], dim=-1)
        kwargs: dict[str, Any] = dict()
        if bboxes._normalized:
            kwargs['normalized'] = True
        if bboxes.has_image_wh:
            kwargs['image_wh'] = bboxes._image_wh
        return cls(from_, **kwargs)

    def to(self, cls: type[T]) -> T:
        return cls.from_(self)

    def intersections(self, other: 'BBoxes') -> torch.Tensor:
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

    def __and__(self, other: 'BBoxes') -> torch.Tensor:
        return self.intersections(other)

    def unions(
        self,
        other: 'BBoxes',
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

    def __or__(self, other: 'BBoxes') -> torch.Tensor:
        r"""Wrap `unions`.

        Args:
            other: :math:`n' \times 4`.

        Returns:
            :math:`n \times n'`.
        """
        intersections = self.intersections(other)
        return self.unions(other, intersections)

    def ious(self, other: 'BBoxes', eps: float = 1e-6) -> torch.Tensor:
        r"""Intersections over unions.

        Args:
            other: :math:`n' \times 4`.
            eps: avoid division by zero.

        Returns:
            :math:`n \times n'`.
        """
        intersections = self.intersections(other)
        unions = self.unions(other, intersections)
        unions = unions.clamp_min_(eps)
        return intersections / unions

    def round(self) -> Self:
        return self.to(BBoxesXYXY).round().to(self.__class__)

    def expand(self, ratio_wh: tuple[float, float]) -> Self:
        return self.to(BBoxesCXCYWH).expand(ratio_wh).to(self.__class__)

    def clamp(self) -> Self:
        return self.to(BBoxesXYXY).clamp().to(self.__class__)

    def indices(
        self,
        *,
        min_area: float | None = None,
        min_wh: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        indices = self._bboxes.new_ones(len(self), dtype=torch.bool)
        if min_area is not None:
            indices &= self.area.ge(min_area)
        if min_wh is not None:
            indices &= self.wh.ge(torch.tensor(min_wh)).all(-1)
        return indices


class BBoxesXY__(BBoxes):

    @property
    def left(self) -> torch.Tensor:
        return self._bboxes[:, 0]

    @property
    def top(self) -> torch.Tensor:
        return self._bboxes[:, 1]

    @property
    def lt(self) -> torch.Tensor:
        return self._bboxes[:, :2]

    @classmethod
    def _from1(cls, bboxes: BBoxes) -> torch.Tensor:
        return bboxes.lt


class BBoxesCXCY__(BBoxes):

    @property
    def center_x(self) -> torch.Tensor:
        return self._bboxes[:, 0]

    @property
    def center_y(self) -> torch.Tensor:
        return self._bboxes[:, 1]

    @property
    def center(self) -> torch.Tensor:
        return self._bboxes[:, :2]

    @classmethod
    def _from1(cls, bboxes: BBoxes) -> torch.Tensor:
        return bboxes.center


class BBoxes__XY(BBoxes):  # noqa: N801

    @property
    def right(self) -> torch.Tensor:
        return self._bboxes[:, 2]

    @property
    def bottom(self) -> torch.Tensor:
        return self._bboxes[:, 3]

    @property
    def rb(self) -> torch.Tensor:
        return self._bboxes[:, 2:]

    @classmethod
    def _from2(cls, bboxes: BBoxes) -> torch.Tensor:
        return bboxes.rb


class BBoxes__WH(BBoxes):  # noqa: N801

    @property
    def width(self) -> torch.Tensor:
        return self._bboxes[:, 2]

    @property
    def height(self) -> torch.Tensor:
        return self._bboxes[:, 3]

    @property
    def wh(self) -> torch.Tensor:
        return self._bboxes[:, 2:]

    @classmethod
    def _from2(cls, bboxes: BBoxes) -> torch.Tensor:
        return bboxes.wh


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

    @staticmethod
    def _denormalized_decorator(func):

        @functools.wraps(func)
        def wrapper(self: Self, *args, **kwargs) -> Self:
            if normalized := self._normalized:
                self = self.denormalize()
            self = func(self, *args, **kwargs)
            if normalized:
                self = self.normalize()
            return self

        return wrapper

    @_denormalized_decorator
    def round(self) -> Self:
        lt = self.lt.floor()
        rb = self.rb.ceil()
        bboxes = torch.cat([lt, rb], dim=-1)
        self = self._copy(bboxes)
        return self

    @_denormalized_decorator
    def clamp(self) -> Self:
        image_wh = self._bboxes.new_tensor(self.image_wh * 2)
        bboxes = self._bboxes.clamp_min(0).clamp_max(image_wh)
        self = self._copy(bboxes)
        return self


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

    def expand(self, ratio_wh: tuple[float, float]) -> Self:
        wh = self.wh * self._bboxes.new_tensor(ratio_wh)
        bboxes = torch.cat([self.center, wh], dim=-1)
        return self._copy(bboxes)
