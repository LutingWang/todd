__all__ = [
    'SpringOpticalFlowAccessLayer',
    'SpringCV2AccessLayer',
]

import pathlib
from pathlib import Path
from typing import Iterator, TypeVar

import numpy as np
import numpy.typing as npt

from todd.datasets.access_layers import CV2AccessLayer, FolderAccessLayer

from ...optical_flow import Flo5OpticalFlow
from ..registries import OFEAccessLayerRegistry
from .optical_flow import OpticalFlowAccessLayer

VT = TypeVar('VT')


class SpringMixin(FolderAccessLayer[VT]):

    def __init__(self, *args, modality: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._modality = modality

    def _files(self) -> Iterator[Path]:
        files = super()._files()
        return filter(lambda file: file.parts[-2] == self._modality, files)

    def _file(self, key: str) -> pathlib.Path:
        scene, frame = key.split('/')
        key = f'{scene}/{self._modality}/{self._modality}_{frame}'
        return super()._file(key)

    def __iter__(self) -> Iterator[str]:
        for key in super().__iter__():
            scene, modality, frame = key.split('/')
            assert modality == self._modality
            assert frame.startswith(self._modality + '_')
            frame = frame.removeprefix(self._modality + '_')
            yield scene + '/' + frame


@OFEAccessLayerRegistry.register_()
class SpringOpticalFlowAccessLayer(
    SpringMixin[Flo5OpticalFlow],
    OpticalFlowAccessLayer[Flo5OpticalFlow],
):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, optical_flow_type=Flo5OpticalFlow, **kwargs)


@OFEAccessLayerRegistry.register_()
class SpringCV2AccessLayer(SpringMixin[npt.NDArray[np.uint8]], CV2AccessLayer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, suffix='png', **kwargs)
