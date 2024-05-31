__all__ = [
    'EMAShadow',
]

import torch

from ...patches.py import classproperty
from ...registries import BuildSpec, BuildSpecMixin
from ...utils import EMA
from ..registries import ShadowRegistry
from .base import BaseShadow


@ShadowRegistry.register_()
class EMAShadow(BuildSpecMixin, BaseShadow):
    """Exponential Moving Average (EMA) Shadow.

    This class represents a shadow model that applies exponential moving
    average to the input data.

    Args:
        ema: The EMA object used for applying exponential moving average.

    A copy of the state dict of the given module is stored as the initial
    shadow:

        >>> import torch.nn as nn
        >>> module = nn.Module()
        >>> module.register_buffer('p', torch.tensor([1., 2., 3.]))
        >>> ema = EMAShadow(module=module, ema=EMA())
        >>> dict(ema.shadow)
        {'p': tensor([1., 2., 3.])}

    The shadow updates according to the model:

        >>> module.register_buffer('p', torch.tensor([4., 5., 6.]))
        >>> ema(module)
        >>> dict(ema.shadow)
        {'p': tensor([1.0300, 2.0300, 3.0300])}
    """

    def __init__(self, *args, ema: EMA, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ema = ema

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec = BuildSpec(ema=lambda c: EMA(**c))
        return super().build_spec | build_spec

    def _forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply exponential moving average to the input data.

        Args:
            x: The input tensor.
            y: The target tensor.

        Returns:
            The output tensor after applying exponential moving average.
        """
        return self._ema(x, y)
