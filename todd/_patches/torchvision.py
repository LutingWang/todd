import enum

import torchvision
import torchvision.transforms as transforms
from packaging import version
from PIL import Image

from .logging import get_logger

_logger = get_logger()

if version.parse(torchvision.__version__) < version.parse('0.9.0'):
    _logger.warning(
        "Monkey patching `torchvision.transforms.InterpolationMode`.",
    )

    class InterpolationMode(enum.Enum):
        BICUBIC = Image.BICUBIC

    transforms.InterpolationMode = InterpolationMode
