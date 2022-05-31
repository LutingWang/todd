from functools import lru_cache
from typing_extensions import Literal
import enum
import functools
import os
import typing
import zipfile

from PIL import Image
import torch
import torchvision.transforms as transforms

from .logger import get_logger
_logger = get_logger()


try:
    from functools import cached_property
except ImportError:
    _logger.warning("Monkey patching `functools.cached_property`.")

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

    functools.cached_property = cached_property

try:
    from functools import cache
except ImportError:
    _logger.warning("Monkey patching `functools.cache`.")

    def cache(user_function):
        return lru_cache(maxsize=None)(user_function)

    functools.cache = cache

try:
    from zipfile import Path
except ImportError:
    _logger.warning("Monkey patching `zipfile.Path`.")

    class Path:
        def __init__(self, root: str, at: str = ''):
            self._root = zipfile.ZipFile(root)
            self._at = at
        
        def _next(self, at: str) -> 'Path':
            return Path(self._root.filename, at)

        def _is_child(self, path: 'Path') -> bool:
            return os.path.dirname(path._at.rstrip("/")) == self._at.rstrip("/")

        @cached_property
        def _namelist(self) -> typing.List[str]:
            return self._root.namelist()

        def read_bytes(self) -> bytes:
            return self._root.read(self._at)
        
        def exists(self) -> bool:
            return self._at in self._namelist

        def iterdir(self) -> typing.Iterator['Path']:
            subs = map(self._next, self._namelist)
            return filter(self._is_child, subs)

        def __str__(self):
            return os.path.join(self._root.filename, self._at)

        def __truediv__(self, at: str) -> 'Path':
            return self._next(os.path.join(self._at, at))

    zipfile.Path = Path

try:
    from torch import maximum, minimum
except ImportError:
    _logger.warning("Monkey patching `torch.maximum` and `torch.minimum`.")
    torch.maximum = torch.max
    torch.Tensor.maximum = torch.Tensor.max
    torch.minimum = torch.min
    torch.Tensor.minimum = torch.Tensor.min

try:
    from typing import Literal
except ImportError:
    _logger.warning("Monkey patching `typing.Literal`.")
    typing.Literal = Literal

try:
    from torchvision.transforms import InterpolationMode
except ImportError:
    _logger.warning("Monkey patching `torchvision.transforms.InterpolationMode`.")

    class InterpolationMode(enum.Enum):
        BICUBIC = Image.BICUBIC

    transforms.InterpolationMode = InterpolationMode


from . import adapts
from . import distillers
from . import hooks
from . import logger
from . import losses
from . import reproduction
from . import schedulers
from . import utils
from . import visuals


__all__ = [
    'adapts', 'distillers', 'hooks', 'logger', 'losses', 'reproduction', 'schedulers', 'utils', 'visuals',
]
