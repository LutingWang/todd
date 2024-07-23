__all__ = [
    'TAPVidDAVISDataset',
]

from typing import TypedDict

import torch

from todd import Config, RegistryMeta
from todd.bases.registries import Item
from todd.datasets import BaseDataset

from .access_layers import TAPVidDAVISAccessLayer
from .access_layers.tap_vid_davis import VT


class T(TypedDict):
    id_: str
    video: torch.Tensor
    query_ts: torch.Tensor
    query_points: torch.Tensor
    target_points: torch.Tensor
    occluded: torch.Tensor


class TAPVidDAVISDataset(BaseDataset[T, str, VT]):

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        access_layer = config.pop('access_layer')
        config = super().build_pre_hook(config, registry, item)
        config.access_layer = access_layer
        return config

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
        id_, vt = self._access(index)
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
            id_=id_,
            video=video,
            query_ts=torch.cat(query_ts),
            query_points=torch.cat(query_points),
            target_points=points,
            occluded=occluded,
        )
