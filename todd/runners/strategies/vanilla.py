__all__ = [
    'VanillaStrategy',
]

import torch.nn as nn

from ...base import StrategyRegistry
from .base import BaseStrategy


@StrategyRegistry.register()
class VanillaStrategy(BaseStrategy):

    @property
    def model(self) -> nn.Module:
        return self._model
