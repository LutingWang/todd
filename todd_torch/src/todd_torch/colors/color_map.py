__all__ = [
    'ColorMap',
]

import cv2
import numpy as np
import numpy.typing as npt
import torch


class ColorMap:

    def __init__(self, *args, color_map: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._color_map = color_map

    def __call__(self, tensor: torch.Tensor) -> npt.NDArray[np.uint8]:
        assert tensor.dim() == 2
        tensor = tensor - tensor.min()
        tensor = tensor / tensor.max()
        tensor = tensor * 255
        tensor = tensor.type(torch.uint8)
        return cv2.applyColorMap(  # type: ignore[return-value]
            tensor.numpy(),
            self._color_map,
        )
