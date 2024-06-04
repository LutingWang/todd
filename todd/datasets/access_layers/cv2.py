__all__ = [
    'CV2AccessLayer',
]

from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from ..registries import AccessLayerRegistry
from .folder import FolderAccessLayer
from .suffix import SuffixMixin

VT = npt.NDArray[np.uint8]


@AccessLayerRegistry.register_()
class CV2AccessLayer(SuffixMixin[VT], FolderAccessLayer[VT]):

    def __getitem__(self, key: str) -> VT:
        return cast(VT, cv2.imread(str(self._file(key))))

    def __setitem__(self, key: str, value: VT) -> None:
        cv2.imwrite(str(self._file(key)), value)
