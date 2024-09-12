__all__ = [
    'get_image',
]

from io import BytesIO

import numpy as np
import numpy.typing as npt
import requests
from PIL import Image


def get_image(url: str) -> npt.NDArray[np.uint8]:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    with Image.open(BytesIO(response.content)) as image:
        image = image.convert('RGB')
        return np.array(image)
