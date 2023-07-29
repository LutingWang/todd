import builtins
import enum

import torch
import torchvision
import torchvision.transforms as transforms
from packaging import version
from PIL import Image

from .logger import logger

try:
    import ipdb
    logger.info("`ipdb` is installed. Using it for debugging.")
    builtins.breakpoint = ipdb.set_trace
except ImportError:
    pass

if torch.__version__ < '1.7.0':
    logger.warning("Monkey patching `torch.maximum` and `torch.minimum`.", )
    torch.maximum = torch.max
    torch.Tensor.maximum = torch.Tensor.max
    torch.minimum = torch.min
    torch.Tensor.minimum = torch.Tensor.min

if version.parse(torchvision.__version__) < version.parse('0.9.0'):
    logger.warning(
        "Monkey patching `torchvision.transforms.InterpolationMode`.",
    )

    class InterpolationMode(enum.Enum):
        BICUBIC = Image.BICUBIC

    transforms.InterpolationMode = InterpolationMode
