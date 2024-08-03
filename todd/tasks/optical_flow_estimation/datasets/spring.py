# pylint: disable=duplicate-code

__all__ = [
    'SpringDataset',
]

from typing import TypedDict

import torch

from todd import Config

from ..optical_flow import Flo5OpticalFlow
from ..registries import OFEDatasetRegistry
from .access_layers import SpringCV2AccessLayer, SpringOpticalFlowAccessLayer
from .base import BaseDataset

VT = Flo5OpticalFlow


class T(TypedDict):
    id_: str
    of: torch.Tensor
    frame1: torch.Tensor
    frame2: torch.Tensor


@OFEDatasetRegistry.register_()
class SpringDataset(BaseDataset[T, VT]):

    def __init__(
        self,
        *args,
        access_layer: Config,
        frame_access_layer: Config | None = None,
        **kwargs,
    ) -> None:
        access_layer.setdefault('subfolder_action', 'walk')
        flo_access_layer = SpringOpticalFlowAccessLayer(**access_layer)
        frame_access_layer = (
            access_layer if frame_access_layer is None else access_layer
            | frame_access_layer
        )
        self._frame = SpringCV2AccessLayer(**frame_access_layer)  # noqa: E501 pylint: disable=abstract-class-instantiated
        super().__init__(*args, access_layer=flo_access_layer, **kwargs)

    def build_keys(self) -> list[str]:
        keys = super().build_keys()
        frame_keys = list(self._frame)
        return [
            key for key in keys
            if key in frame_keys and self._next_key(key) in frame_keys
        ]

    def _next_key(self, key: str) -> str:
        scene, frame = key.rsplit('/', 1)
        frame = f'{int(frame) + 1:04d}'
        return scene + '/' + frame

    def __getitem__(self, index: int) -> T:
        key, of = self._access(index)
        return T(
            id_=key,
            of=of.to_tensor(),
            frame1=torch.tensor(self._frame[key]),
            frame2=torch.tensor(self._frame[self._next_key(key)]),
        )
