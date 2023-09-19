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
]

from abc import ABC, abstractmethod
from typing import Generator, TypeVar
from typing_extensions import Self

import einops
import torch

BBox = tuple[float, float, float, float]
T = TypeVar('T', bound='BBoxes')


class BBoxes(ABC):

    def __init__(self, bboxes: torch.Tensor) -> None:
        """Initialize.

        Args:
            bboxes: :math:`n \\times 4`.

        Bounding boxes must be 2 dimensional and the second dimension must be
        of size 4:

            >>> bboxes = torch.tensor([[10.0, 20.0, 40.0, 100.0]])
            >>> BBoxesXYXY(bboxes[0])
            Traceback (most recent call last):
            ...
            ValueError: bboxes must be at least 2-dim
            >>> BBoxesXYXY(bboxes[:, :3])
            Traceback (most recent call last):
            ...
            ValueError: bboxes must have 4 columns
        """
        if bboxes.ndim < 2:
            raise ValueError('bboxes must be at least 2-dim')
        if bboxes.shape[-1] != 4:
            raise ValueError('bboxes must have 4 columns')
        self._bboxes = bboxes

    def __len__(self) -> int:
        """Number of bboxes.

        Examples:

            >>> bboxes = torch.tensor([
            ...     [5.0, 15.0, 8.0, 18.0],
            ...     [5.0, 15.0, 8.0, 60.0],
            ... ])
            >>> len(BBoxesXYXY(bboxes))
            2
        """
        return self._bboxes.shape[0]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._bboxes})'

    def __add__(self, other: Self) -> Self:
        """Concatenate bboxes.

        Args:
            other: :math:`n' \\times 4`.

        Returns:
            :math:`(n + n') \\times 4`, where `n` is the length of `self`.

        Examples:

            >>> a = torch.tensor([[5.0, 15.0, 8.0, 18.0]])
            >>> b = torch.tensor([[5.0, 15.0, 8.0, 60.0]])
            >>> BBoxesXYXY(a) + BBoxesXYXY(b)
            BBoxesXYXY(tensor([[ 5., 15.,  8., 18.],
                    [ 5., 15.,  8., 60.]]))
        """
        bboxes = torch.cat([self._bboxes, other._bboxes])
        return self.__class__(bboxes)

    def __getitem__(self, indices) -> Self:
        """Get specific bboxes.

        Args:
            indices: a index or multiple indices.

        Returns:
            If `indices` refers to a single box, return a ``tuple``.
            Otherwise, return ``BBoxes``.

        Examples:

            >>> bboxes = torch.tensor([
            ...     [5.0, 15.0, 8.0, 18.0],
            ...     [5.0, 15.0, 8.0, 60.0],
            ...     [5.0, 15.0, 8.0, 105.0],
            ... ])
            >>> BBoxesXYXY(bboxes)[0]
            BBoxesXYXY(tensor([[ 5., 15.,  8., 18.]]))
            >>> BBoxesXYXY(bboxes)[:-1]
            BBoxesXYXY(tensor([[ 5., 15.,  8., 18.],
                    [ 5., 15.,  8., 60.]]))
        """
        tensor = self._bboxes[indices]
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return self.__class__(tensor)

    def __iter__(self) -> Generator[BBox, None, None]:
        """Iterate over bboxes.

        Yields:
            One bbox.

        Examples:

            >>> bboxes = torch.tensor([
            ...     [5.0, 15.0, 8.0, 18.0],
            ...     [5.0, 15.0, 8.0, 60.0],
            ...     [5.0, 15.0, 8.0, 105.0],
            ... ])
            >>> for bbox in BBoxesXYXY(bboxes):
            ...     print(bbox)
            (5.0, 15.0, 8.0, 18.0)
            (5.0, 15.0, 8.0, 60.0)
            (5.0, 15.0, 8.0, 105.0)
        """
        yield from map(tuple, self._bboxes.tolist())

    def __and__(self, other: 'BBoxes') -> torch.Tensor:
        """Intersections.

        Args:
            other: :math:`n' \\times 4`.

        Returns:
            :math:`n \\times n'`.
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

    def __or__(self, other: 'BBoxes') -> torch.Tensor:
        """Wraps `unions`.

        Args:
            other: :math:`n' \\times 4`.

        Returns:
            :math:`n \\times n'`.
        """
        return self.unions(other, self & other)

    @classmethod
    @abstractmethod
    def _from1(cls, bboxes: 'BBoxes') -> torch.Tensor:
        pass

    @classmethod
    @abstractmethod
    def _from2(cls, bboxes: 'BBoxes') -> torch.Tensor:
        pass

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

    @abstractmethod
    def _translate1(self, offset: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _translate2(self, offset: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def area(self) -> torch.Tensor:
        return self.width * self.height

    def to_tensor(self) -> torch.Tensor:
        return self._bboxes

    def to(self, cls: type[T]) -> T:
        return cls.from_(self)

    def unions(
        self,
        other: 'BBoxes',
        intersections: torch.Tensor,
    ) -> torch.Tensor:
        """Unions.

        Args:
            other: :math:`n' \\times 4`.
            intersections: :math:`n \\times n'`

        Returns:
            :math:`n \\times n'`.
        """
        return self.area[:, None] + other.area[None, :] - intersections

    def ious(self, other: 'BBoxes', eps: float = 1e-6) -> torch.Tensor:
        """Intersections over unions.

        Args:
            other: :math:`n' \\times 4`.
            eps: avoid division by zero.

        Returns:
            :math:`n \\times n'`.
        """
        intersections = self & other
        unions = self.unions(other, intersections).clamp_min_(eps)
        return intersections / unions

    @classmethod
    def from_(cls, bboxes: 'BBoxes') -> Self:
        from1 = cls._from1(bboxes)
        from2 = cls._from2(bboxes)
        bboxes = torch.cat([from1, from2], dim=-1)
        return cls(bboxes)

    def round(self) -> Self:
        return self.to(BBoxesXYXY).round().to(self.__class__)

    def expand(self, ratio_wh: tuple[float, float]) -> Self:
        return self.to(BBoxesCXCYWH).expand(ratio_wh).to(self.__class__)

    def clamp(self, image_wh: tuple[int, int]) -> Self:
        return self.to(BBoxesXYXY).clamp(image_wh).to(self.__class__)

    def scale(self, ratio_wh: tuple[float, float]) -> Self:
        scale = torch.tensor(ratio_wh * 2)
        bboxes = self._bboxes * scale
        return self.__class__(bboxes)

    def translate(self, offset: torch.Tensor) -> Self:
        offset1 = self._translate1(offset)
        offset2 = self._translate2(offset)
        offset = torch.cat([offset1, offset2], dim=-1)
        bboxes = self._bboxes + offset
        return self.__class__(bboxes)

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

    def _translate1(self, offset: torch.Tensor) -> torch.Tensor:
        return offset


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

    def _translate1(self, offset: torch.Tensor) -> torch.Tensor:
        return offset


class BBoxes__XY(BBoxes):

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

    def _translate2(self, offset: torch.Tensor) -> torch.Tensor:
        return offset


class BBoxes__WH(BBoxes):

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

    def _translate2(self, offset: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(offset)


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

    def round(self) -> Self:
        lt = self.lt.floor()
        rb = self.rb.ceil()
        bboxes = torch.cat([lt, rb], dim=-1)
        return self.__class__(bboxes)

    def clamp(self, image_wh: tuple[int, int]) -> Self:
        lt = self.lt.clamp_min(0)
        rb = self.rb.clamp_max(torch.tensor(image_wh))
        bboxes = torch.cat([lt, rb], dim=-1)
        return self.__class__(bboxes)


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
        wh = self.wh * torch.tensor(ratio_wh)
        bboxes = torch.cat([self.center, wh], dim=-1)
        return self.__class__(bboxes)
