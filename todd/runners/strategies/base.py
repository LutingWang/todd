__all__ = ["BaseStrategy"]

import torch.nn as nn

from ...base import StrategyRegistry


@StrategyRegistry.register()
class BaseStrategy:

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def get_model(self, model: nn.Module) -> nn.Module:
        return model
