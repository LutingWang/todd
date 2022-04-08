import enum
import functools
import typing
from functools import lru_cache
from typing_extensions import Literal

import torch
import torchvision.transforms as transforms
from PIL import Image


try:
    from functools import cached_property
except ImportError:

    class cached_property:
        def __init__(self, func):
            self.func = func
            self.__doc__ = func.__doc__
    
        def __set_name__(self, owner, name):
            self.attrname = name
    
        def __get__(self, instance, owner=None):
            if self.attrname in instance.__dict__:
                return instance.__dict__[self.attrname]
            val = self.func(instance)
            instance.__dict__[self.attrname] = val
            return val

    functools.cached_property = property

try:
    from functools import cache
except ImportError:
    def cache(user_function):
        return lru_cache(maxsize=None)(user_function)

    functools.cache = cache

try:
    from torch import maximum, minimum
except ImportError:
    torch.maximum = torch.max
    torch.Tensor.maximum = torch.Tensor.max
    torch.minimum = torch.min
    torch.Tensor.minimum = torch.Tensor.min

try:
    from typing import Literal
except ImportError:
    typing.Literal = Literal

try:
    from torchvision.transforms import InterpolationMode
except ImportError:

    class InterpolationMode(enum.Enum):
        BICUBIC = Image.BICUBIC

    transforms.InterpolationMode = InterpolationMode


from . import adapts
from . import distillers
from . import hooks
from . import logger
from . import losses
from . import utils
from . import schedulers
from . import visuals


__all__ = [
    'adapts', 'distillers', 'hooks', 'logger', 'losses', 'utils', 'schedulers', 'visuals',
]
