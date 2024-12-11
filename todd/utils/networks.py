__all__ = [
    'get_bytes',
    'get_image',
    'get_audio',
]

from io import BytesIO

import numpy as np
import numpy.typing as npt
import requests
import soundfile as sf
from PIL import Image


def get_bytes(url: str) -> BytesIO:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return BytesIO(response.content)


def get_image(url: str) -> npt.NDArray[np.uint8]:
    with Image.open(get_bytes(url)) as image:
        image = image.convert('RGB')
        return np.array(image)


def get_audio(url: str) -> tuple[npt.NDArray[np.float64], int]:
    return sf.read(get_bytes(url))
