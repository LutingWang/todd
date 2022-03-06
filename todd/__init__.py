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
            cache[self.attrname] = val
            return val

    import functools
    functools.cached_property = property

try:
    from torch import maximum, minimum
except ImportError:
    import torch
    torch.maximum = torch.max
    torch.Tensor.maximum = torch.Tensor.max
    torch.minimum = torch.min
    torch.Tensor.minimum = torch.Tensor.min

try:
    from typing import Literal
except ImportError:
    import typing
    from typing_extensions import Literal
    typing.Literal = Literal

from . import adapts
from . import distillers
from . import hooks
from . import losses
from . import utils
from . import schedulers
from . import visuals


__all__ = [
    'adapts', 'distillers', 'hooks', 'losses', 'utils', 'schedulers', 'visuals',
]


from mmcv.runner import BaseModule
import torch.nn as nn
class ModuleDict(BaseModule, nn.ModuleDict):
    """ModuleDict in openmmlab.
    Args:
        modules (dict, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module).
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules)
import mmcv.runner
mmcv.runner.ModuleDict = ModuleDict