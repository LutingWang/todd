__all__ = [
    'InitWeightsMixin',
]

from abc import ABC

from torch import nn

from ..base import Config


class InitWeightsMixin(nn.Module, ABC):

    def init_weights(self, config: Config) -> bool:
        if hasattr(super(), 'init_weights'):
            return super().init_weights(config)  # type: ignore[misc]
        return True
