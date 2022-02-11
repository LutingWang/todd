try:
    from functools import cached_property
except ImportError:
    import functools
    functools.cached_property = property

try:
    from torch import maximum
    from torch import minimum
except ImportError:
    import torch
    torch.maximum = torch.max
    torch.Tensor.maximum = torch.Tensor.max
    torch.minimum = torch.min
    torch.Tensor.minimum = torch.Tensor.min

try:
    from typing import Literal, TypeAlias
except ImportError:
    import typing
    from typing_extensions import Literal, TypeAlias
    typing.Literal = Literal
    typing.TypeAlias = TypeAlias

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
