__all__ = [
    'CV2AccessLayer',
]

from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from ..registries import AccessLayerRegistry
from .file import FileAccessLayer
from .suffix import SuffixMixin

V = npt.NDArray[np.uint8]


@AccessLayerRegistry.register_()
class CV2AccessLayer(SuffixMixin[V], FileAccessLayer[V]):

    def __getitem__(self, key: str) -> V:
        image = cv2.imread(str(self._path(key)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cast(V, image)

    def __setitem__(self, key: str, value: V) -> None:
        image = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self._path(key)), image)
