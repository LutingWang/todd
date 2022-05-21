from functools import cached_property
from typing import Any, Optional, Tuple, TypeVar
import warnings

import numpy as np
import torch


def iou(bboxes1: torch.Tensor, bboxes2: Optional[torch.Tensor] = None, eps: float = 1e-6):
    warnings.warn("iou is deprecated, use BBoxes.ious instead.")
    return BBoxes(bboxes1).ious(BBoxes(bboxes2), eps)


T = TypeVar('T', np.ndarray, torch.Tensor)


# TODO: add support for numpy
class BBoxes:
    def __init__(self, bboxes: T):
        """
        Args:
            bboxes: * x 4
                (x1, y1, x2, y2)
        """
        self._bboxes = bboxes

    def to_tensor(self) -> torch.Tensor:
        if isinstance(self._bboxes, torch.Tensor):
            return self._bboxes
        return torch.as_tensor(self._bboxes)

    @property
    def empty(self) -> bool:
        return self._bboxes.shape[0] == 0

    @cached_property
    def shapes(self) -> T:
        return self._bboxes[..., 2:] - self._bboxes[..., :2]

    @cached_property
    def areas(self) -> T:
        return self.shapes[..., 0] * self.shapes[..., 1]

    def __add__(self, other: 'BBoxes') -> 'BBoxes':
        """
        Args:
            other: *1 x 4
        
        Returns:
            ious: (* + *1) x 4
        """
        assert isinstance(other, self.__class__)
        bboxes = torch.cat([self.to_tensor(), other.to_tensor()])
        return self.__class__(bboxes)


    def __getitem__(self, index: Any) -> 'BBoxes':
        bboxes = self.to_tensor()
        return self.__class__(bboxes[index])

    @staticmethod
    def _xyxy(bboxes: torch.Tensor) -> torch.Tensor:
        return bboxes.clone()

    def xyxy(self) -> 'BBoxes':
        bboxes = self._xyxy(self._bboxes)
        return BBoxes(bboxes)

    @staticmethod
    def _xywh(bboxes: torch.Tensor) -> torch.Tensor:
        bboxes = bboxes.clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        return bboxes

    def xywh(self) -> 'BBoxesXYWH':
        bboxes = self._xywh(self._bboxes)
        return BBoxesXYWH(bboxes)

    @staticmethod
    def _round(bboxes: torch.Tensor) -> torch.Tensor:
        bboxes = bboxes.clone()
        bboxes[:, :2].floor_()
        bboxes[:, 2:].ceil_()
        return bboxes

    def round(self) -> 'BBoxes':
        bboxes = self._round(self._bboxes)
        return BBoxes(bboxes)

    def intersections(self, other: 'BBoxes') -> torch.Tensor:
        """
        Args:
            other: *1 x 4
        
        Returns:
            intersections: * x *1
        """
        lt = torch.maximum(  # [*, *1, 2]
            self._bboxes[:, None, :2],
            other._bboxes[None, :, :2],
        )
        rb = torch.minimum(  # [*, *1, 2]
            self._bboxes[:, None, 2:],
            other._bboxes[None, :, 2:],
        )
        wh = (rb - lt).clamp_min_(0)
        return wh[..., 0] * wh[..., 1]

    def _unions(self, other: 'BBoxes', intersections: torch.Tensor) -> torch.Tensor:
        """
        Args:
            other: *1 x 4
            intersections: * x *1
            
        Returns:
            unions: * x *1
        """
        area1 = self.areas
        area2 = other.areas
        return area1[:, None] + area2[None, :] - intersections

    def unions(self, other: 'BBoxes') -> torch.Tensor:
        """
        Args:
            other: *1 x 4
        
        Returns:
            unions: * x *1
        """
        intersections = self.intersections(other)
        return self._unions(other, intersections)

    def ious(self, other: 'BBoxes', eps: float = 1e-6) -> torch.Tensor:
        """
        Args:
            other: *1 x 4
        
        Returns:
            ious: * x *1
        """
        if self.empty or other.empty:
            return self._bboxes.new_empty((self._bboxes.shape[0], other._bboxes.shape[0]))

        intersections = self.intersections(other)
        unions = self._unions(other, intersections)
        unions = unions.clamp_min_(eps)
        ious = intersections / unions
        return ious

    def expand(self, ratio: float, image_shape: Optional[Tuple[int]] = None) -> 'BBoxes':
        """
        Args:
            bboxes: * x 4
        
        Returns:
            expanded_bboxes: * x 4
        """
        offsets = self.shapes * (ratio - 1) / 2
        bboxes = self._bboxes.clone()
        bboxes[..., :2] -= offsets
        bboxes[..., 2:] += offsets

        bboxes[..., :2].clamp_min_(0)
        if image_shape is not None:
            h, w = image_shape
            bboxes[..., 2].clamp_max_(w)
            bboxes[..., 3].clamp_max_(h)
        
        return self.__class__(bboxes)


class BBoxesXYWH(BBoxes):
    def __init__(self, bboxes: torch.Tensor):
        bboxes = bboxes.clone()
        bboxes[:, 2:] += bboxes[:, :2]
        super().__init__(bboxes)

    def to_tensor(self) -> torch.Tensor:
        bboxes = super().to_tensor()
        return self._xywh(bboxes)

    def round(self) -> 'BBoxesXYWH':
        bboxes = self._round(self._bboxes)
        bboxes = self._xywh(bboxes)
        return BBoxesXYWH(bboxes)
