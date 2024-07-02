__all__ = [
    'SintelDataset',
]

from typing import TypedDict

import torch

from todd import Config
from todd.datasets.access_layers import CV2AccessLayer

from ..optical_flow import FloOpticalFlow
from ..registries import OFEDatasetRegistry
from .access_layers import OpticalFlowAccessLayer
from .base import BaseDataset

VT = FloOpticalFlow


class T(TypedDict):
    id_: str
    of: torch.Tensor
    frame1: torch.Tensor
    frame2: torch.Tensor
    invalid: torch.Tensor
    occlusion: torch.Tensor


@OFEDatasetRegistry.register_()
class SintelDataset(BaseDataset[T, VT]):

    def __init__(
        self,
        *args,
        access_layer: Config,
        pass_: str,
        **kwargs,
    ) -> None:
        task_name = 'training/'
        flo_access_layer = OpticalFlowAccessLayer(
            **access_layer,
            task_name=task_name + 'flow',
            optical_flow_type=VT,
            subfolder_action='walk',
        )
        super().__init__(*args, access_layer=flo_access_layer, **kwargs)
        self._frame = CV2AccessLayer(
            **access_layer,
            task_name=task_name + pass_,
            suffix='png',
        )
        self._invalid = CV2AccessLayer(
            **access_layer,
            task_name=task_name + 'invalid',
            suffix='png',
        )
        self._occlusion = CV2AccessLayer(
            **access_layer,
            task_name=task_name + 'occlusions',
            suffix='png',
        )

    def _next_key(self, key: str, prefix: str = 'frame_') -> str:
        scene, frame = key.split('/')
        assert frame.startswith(prefix)
        frame = frame.removeprefix(prefix)
        frame = f'{int(frame) + 1:04d}'
        frame = prefix + frame
        return scene + '/' + frame

    def __getitem__(self, index: int) -> T:
        key, of = self._access(index)
        return T(
            id_=key,
            of=of.to_tensor(),
            frame1=torch.tensor(self._frame[key]),
            frame2=torch.tensor(self._frame[self._next_key(key)]),
            invalid=torch.tensor(self._invalid[key]),
            occlusion=torch.tensor(self._occlusion[key]),
        )
