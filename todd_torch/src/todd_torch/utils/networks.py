__all__ = [
    'get_image',
    'get_audio',
]

import numpy as np
import numpy.typing as npt
import torch
import torchaudio
from PIL import Image

from todd.utils import get_bytes


def get_image(url: str) -> npt.NDArray[np.uint8]:
    with Image.open(get_bytes(url)) as image:
        image = image.convert('RGB')
        return np.array(image)


def get_audio(url: str) -> tuple[torch.Tensor, int]:
    return torchaudio.load(get_bytes(url))
