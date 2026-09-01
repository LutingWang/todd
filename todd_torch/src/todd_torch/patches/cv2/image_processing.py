__all__ = [
    'ColorMap',
]

import cv2
import numpy as np
import numpy.typing as npt
import torch

# https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc.hpp


class ColorMap:

    def __init__(self, color_map: int) -> None:
        self._color_map = color_map

    def __call__(self, tensor: torch.Tensor) -> npt.NDArray[np.uint8]:
        assert tensor.dim() == 2
        tensor = tensor - tensor.min()
        tensor = tensor / tensor.max()
        tensor = tensor * 255
        tensor = tensor.type(torch.uint8)
        return cv2.applyColorMap(tensor.numpy(), self._color_map)
