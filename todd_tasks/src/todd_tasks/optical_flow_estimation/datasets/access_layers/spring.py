__all__ = [
    'SpringOpticalFlowAccessLayer',
    'SpringCV2AccessLayer',
]

from abc import ABC
from pathlib import Path
from typing import Iterator, TypeVar

import numpy as np
import numpy.typing as npt

from todd_torch.datasets.access_layers import CV2AccessLayer, FileAccessLayer

from ...optical_flow import Flo5OpticalFlow
from ..registries import OFEAccessLayerRegistry
from .optical_flow import OpticalFlowAccessLayer

V = TypeVar('V')


class SpringMixin(FileAccessLayer[V], ABC):

    def __init__(self, *args, modality: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._modality = modality

    def _paths(self) -> Iterator[Path]:
        paths = super()._paths()
        return filter(lambda path: path.parts[-2] == self._modality, paths)

    def _path(self, key: str) -> Path:
        scene, frame = key.split('/')
        key = f'{scene}/{self._modality}/{self._modality}_{frame}'
        return super()._path(key)

    def __iter__(self) -> Iterator[str]:
        for key in super().__iter__():
            scene, modality, frame = key.split('/')
            assert modality == self._modality
            assert frame.startswith(self._modality + '_')
            frame = frame.removeprefix(self._modality + '_')
            yield scene + '/' + frame


@OFEAccessLayerRegistry.register_()
class SpringOpticalFlowAccessLayer(
    OpticalFlowAccessLayer[Flo5OpticalFlow],
    SpringMixin[Flo5OpticalFlow],
):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, optical_flow_type=Flo5OpticalFlow, **kwargs)


@OFEAccessLayerRegistry.register_()
class SpringCV2AccessLayer(SpringMixin[npt.NDArray[np.uint8]], CV2AccessLayer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, suffix='.png', **kwargs)
