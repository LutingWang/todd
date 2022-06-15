from abc import abstractmethod
from typing import (
    List,
    NamedTuple,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import einops
import numpy as np
import torch

__all__ = [
    'BBoxesXYXY',
    'BBoxesXYWH',
]

T = TypeVar('T', bound='BBoxes')


class BBoxes(Sized):

    def __init__(self, bboxes: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Args:
            bboxes: \\* x 4
                (x1, y1, x2, y2)
        """
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes)
        if bboxes.numel() == 0:
            raise ValueError('bboxes is empty')
        if bboxes.ndim < 2:
            raise ValueError('bboxes must be at least 2-dim')
        if bboxes.shape[-1] != 4:
            raise ValueError('bboxes must have 4 columns')
        self._bboxes = bboxes

    def __len__(self) -> int:
        return self._bboxes.shape[0]

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

    def to_tensor(self) -> torch.Tensor:
        return self._bboxes

    @property
    def shapes(self) -> torch.Tensor:
        return self.wh

    @property
    def areas(self) -> torch.Tensor:
        return self.width * self.height

    @classmethod
    def _cat(cls: Type[T], a: T, b: T) -> T:
        """
        Args:
            a: *1 x 4
            b: *2 x 4

        Returns:
            ious: (*1 + *2) x 4
        """
        bboxes = torch.cat([a.to_tensor(), b.to_tensor()])
        return cls(bboxes)

    def cat(self: T, other: T) -> T:
        return self._cat(self, other)

    def __add__(self: T, other: T) -> T:
        return self.cat(other)

    @classmethod
    def _select(
        cls: Type[T],
        bboxes: T,
        indices: Union[int, slice, torch.Tensor, List[int], Tuple[int, ...]],
    ) -> T:
        """
        Args:
            bboxes: n x 4
            indices: m

        Returns:
            bboxes: m x 4
        """
        return cls(bboxes.to_tensor()[indices])

    def select(
        self: T,
        indices: Union[int, slice, torch.Tensor, List[int], Tuple[int, ...]],
    ) -> T:
        return self._select(self, indices)

    def __getitem__(
        self: T,
        indices: Union[int, slice, torch.Tensor, List[int], Tuple[int, ...]],
    ) -> T:
        return self.select(indices)

    @classmethod
    def _intersections(cls: Type[T], a: T, b: T) -> torch.Tensor:
        """
        Args:
            a: *1 x 4
            b: *2 x 4

        Returns:
            intersections: *1 x *2
        """
        lt = torch.maximum(  # [*1, *2, 2]
            einops.rearrange(a.lt, 'n1 lt -> n1 1 lt'),
            einops.rearrange(b.lt, 'n2 lt -> 1 n2 lt'),
        )
        rb = torch.minimum(  # [*1, *2, 2]
            einops.rearrange(a.rb, 'n1 rb -> n1 1 rb'),
            einops.rearrange(b.rb, 'n2 rb -> 1 n2 rb'),
        )
        wh = rb - lt
        wh = wh.clamp_min_(0)
        return wh[..., 0] * wh[..., 1]

    def intersections(self: T, other: T) -> torch.Tensor:
        return self._intersections(self, other)

    def __and__(self: T, other: T) -> torch.Tensor:
        return self.intersections(other)

    @classmethod
    def _unions(
        cls: Type[T],
        a: T,
        b: T,
        intersections: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            a: *1 x 4
            b: *2 x 4
            intersections: *1 x *2

        Returns:
            unions: *1 x *2
        """
        if intersections is None:
            intersections = cls.intersections(a, b)
        return a.areas[:, None] + b.areas[None, :] - intersections

    def unions(
        self: T,
        other: T,
        intersections: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._unions(self, other, intersections)

    def __or__(self: T, other: T) -> torch.Tensor:
        return self.unions(other)

    def ious(self: T, other: T, eps: float = 1e-6) -> torch.Tensor:
        """
        Args:
            a: *1 x 4
            b: *2 x 4

        Returns:
            ious: *1 x *2
        """
        intersections = self & other
        unions = self.unions(other, intersections).clamp_min_(eps)
        ious = intersections / unions
        return ious

    @classmethod
    @abstractmethod
    def _round(cls: Type[T], bboxes: T) -> T:
        pass

    def round(self: T) -> T:
        return self._round(self)

    @classmethod
    @abstractmethod
    def _expand(
        cls: Type[T],
        bboxes: T,
        ratio: float,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> T:
        """
        Args:
            bboxes: n x 4

        Returns:
            expanded_bboxes: n x 4
        """
        pass

    def expand(
        self: T,
        ratio: float,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> T:
        return self._expand(self, ratio, image_shape)


T_XY = TypeVar('T_XY', bound='BBoxesXY')


class BBoxesXY(BBoxes):
    _BBoxType = NamedTuple(
        '_BBoxType',
        [('lt', torch.Tensor), ('rb', torch.Tensor)],
    )

    @property
    def left(self) -> torch.Tensor:
        return self._bboxes[:, 0]

    @property
    def top(self) -> torch.Tensor:
        return self._bboxes[:, 1]

    @property
    def lt(self) -> torch.Tensor:
        return self._bboxes[:, :2]

    def _round_lt_rb(self) -> 'BBoxesXY._BBoxType':
        lt = self.lt.floor()
        rb = self.rb.ceil()
        return self._BBoxType(lt, rb)

    def _expand_lt_rb(
        self,
        ratio: float,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> 'BBoxesXY._BBoxType':
        offsets = self.wh * (ratio - 1) / 2
        lt = self.lt - offsets
        rb = self.rb + offsets

        lt.clamp_min_(0)
        if image_shape is not None:
            h, w = image_shape
            rb[:, 0].clamp_max_(w)
            rb[:, 1].clamp_max_(h)

        return self._BBoxType(lt, rb)


class BBoxesXYXY(BBoxesXY):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.wh.gt(0).all()

    @property
    def right(self) -> torch.Tensor:
        return self._bboxes[:, 2]

    @property
    def bottom(self) -> torch.Tensor:
        return self._bboxes[:, 3]

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
    def rb(self) -> torch.Tensor:
        return self._bboxes[:, 2:]

    @property
    def wh(self) -> torch.Tensor:
        return self.rb - self.lt

    @property
    def center(self) -> torch.Tensor:
        return (self.lt + self.rb) / 2

    @classmethod
    def _round(cls: Type[T_XY], bboxes: T_XY) -> T_XY:
        lt, rb = bboxes._round_lt_rb()
        rounded_bboxes = torch.cat([lt, rb], dim=-1)
        return cls(rounded_bboxes)

    @classmethod
    def _expand(
        cls: Type[T_XY],
        bboxes: T_XY,
        ratio: float,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> T_XY:
        lt, rb = bboxes._expand_lt_rb(ratio, image_shape)
        expanded_bboxes = torch.cat([lt, rb], dim=-1)
        return cls(expanded_bboxes)


class BBoxesXYWH(BBoxesXY):

    @property
    def width(self) -> torch.Tensor:
        return self._bboxes[:, 2]

    @property
    def height(self) -> torch.Tensor:
        return self._bboxes[:, 3]

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
    def wh(self) -> torch.Tensor:
        return self._bboxes[:, 2:]

    @property
    def rb(self) -> torch.Tensor:
        return self.lt + self.wh

    @property
    def center(self) -> torch.Tensor:
        return self.lt + self.wh / 2

    @classmethod
    def _round(cls: Type[T_XY], bboxes: T_XY) -> T_XY:
        lt, rb = bboxes._round_lt_rb()
        rounded_bboxes = torch.cat([lt, rb - lt], dim=-1)
        return cls(rounded_bboxes)

    @classmethod
    def _expand(
        cls: Type[T_XY],
        bboxes: T_XY,
        ratio: float,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> T_XY:
        lt, rb = bboxes._expand_lt_rb(ratio, image_shape)
        expanded_bboxes = torch.cat([lt, rb - lt], dim=-1)
        return cls(expanded_bboxes)
