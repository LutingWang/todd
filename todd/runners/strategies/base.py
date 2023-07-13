__all__ = [
    'BaseStrategy',
]

from abc import ABC, abstractmethod

import torch.nn as nn

from ...base import Config, ModelRegistry, StrategyRegistry


@StrategyRegistry.register()
class BaseStrategy(nn.Module, ABC):

    def __init__(self, *args, model: nn.Module | Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(model, Config):
            model = ModelRegistry.build(model)
        self._model: nn.Module = model

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        pass

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)
