__all__ = [
    'TAPVidDAVISDataset',
]

from typing import TypedDict

import torch

from todd.configs import Config
from todd.datasets import BaseDataset
from todd.patches.py import classproperty
from todd.registries import BuildSpec

from .access_layers import VT, TAPVidDAVISAccessLayer


class T(TypedDict):
    video: torch.Tensor
    query_ts: torch.Tensor
    query_points: torch.Tensor
    target_points: torch.Tensor
    occluded: torch.Tensor


class TAPVidDAVISDataset(BaseDataset[T, str, VT]):

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec: BuildSpec = super().build_spec
        build_spec.pop('access_layer')  # pylint: disable=no-member
        return build_spec

    def __init__(  # pylint: disable=useless-super-delegation
        self,
        *args,
        access_layer: Config,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            access_layer=TAPVidDAVISAccessLayer(**access_layer),
            **kwargs,
        )

    def __getitem__(self, index: int) -> T:
        vt = self._access(index)
        video = torch.from_numpy(vt['video'])
        points = torch.from_numpy(vt['points'])
        occluded = torch.from_numpy(vt['occluded'])

        visible = ~occluded

        valid = visible.any(-1)
        points = points[valid]
        occluded = occluded[valid]

        query_ts = []
        query_points = []
        for points_, visible_ in zip(points, visible):
            t = visible_.nonzero()[0]
            query_ts.append(t)
            point = points_[t]
            query_points.append(point)

        return T(
            video=video,
            query_ts=torch.cat(query_ts),
            query_points=torch.cat(query_points),
            target_points=points,
            occluded=occluded,
        )
