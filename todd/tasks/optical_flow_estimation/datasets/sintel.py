__all__ = [
    'SintelDataset',
]

from typing import Any

import torch

from ....configs import Config
from ....datasets import BaseDataset
from ....datasets.access_layers import CV2AccessLayer
from ....patches.py import classproperty
from ....registries import BuildSpec
from ..optical_flow import FloOpticalFlow
from ..registries import OFEDatasetRegistry
from .access_layers import FloAccessLayer

T = dict[str, Any]


@OFEDatasetRegistry.register_()
class SintelDataset(BaseDataset[T, str, FloOpticalFlow]):

    def __init__(
        self,
        *args,
        access_layer: Config,
        pass_: str,
        **kwargs,
    ) -> None:
        task_name = 'training/'
        flo_access_layer = FloAccessLayer(
            **access_layer,
            task_name=task_name + 'flow',
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

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec: BuildSpec = super().build_spec
        build_spec.pop('access_layer')
        return build_spec

    def _next_key(self, key: str, prefix: str = 'frame_') -> str:
        scene, frame = key.split('/')
        assert frame.startswith(prefix)
        frame = frame.removeprefix(prefix)
        frame = f'{int(frame) + 1:04d}'
        frame = prefix + frame
        return scene + '/' + frame

    def __getitem__(self, index: int) -> T:
        key = self._keys[index]
        return dict(
            id_=key,
            of=self._access_layer[key].to_tensor(),
            frame1=torch.tensor(self._frame[key]),
            frame2=torch.tensor(self._frame[self._next_key(key)]),
            invalid=torch.tensor(self._invalid[key]),
            occlusion=torch.tensor(self._occlusion[key]),
        )
