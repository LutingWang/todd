try:
    from functools import cached_property
except ImportError:
    import functools
    functools.cached_property = property

try:
    from torch import maximum
except ImportError:
    import torch
    torch.maximum = torch.max
    torch.Tensor.maximum = torch.Tensor.max

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
