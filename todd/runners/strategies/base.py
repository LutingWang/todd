__all__ = ["BaseStrategy"]

from typing import Any

import torch.nn as nn

from ...base import StrategyRegistry


@StrategyRegistry.register()
class BaseStrategy:

    def setup(self) -> None:
        pass

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def get_model(self, model: nn.Module) -> nn.Module:
        return model

    def state_dict(
        self,
        model: nn.Module,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        return model.state_dict(*args, **kwargs)

    def load_state_dict(
        self,
        model: nn.Module,
        state_dict: dict[str, Any],
        *args,
        **kwargs,
    ) -> None:
        model.load_state_dict(state_dict, *args, **kwargs)
