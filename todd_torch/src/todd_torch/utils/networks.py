__all__ = [
    'get_image',
    'get_audio',
    'image_to_data_url',
]

import base64
import io
import mimetypes

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


def image_to_data_url(
    image: npt.NDArray[np.uint8],
    suffix: str = '.png',
) -> str:
    image_ = Image.fromarray(image)
    with io.BytesIO() as f:
        image_.save(f, Image.registered_extensions()[suffix])
        data = base64.b64encode(f.getvalue()).decode('ascii')
    return f'data:{mimetypes.types_map[suffix]};base64,{data}'
