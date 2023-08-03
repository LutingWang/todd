__all__ = [
    'DDPStrategy',
]

from typing import TYPE_CHECKING

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ...base import Config, StrategyRegistry
from .cuda import CUDAStrategy


@StrategyRegistry.register()
class DDPStrategy(CUDAStrategy):
    _model: DDP

    def __init__(
        self,
        *args,
        wrap_model: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if wrap_model is None:
            wrap_model = Config()
        self._wrap_model(wrap_model)

    def _wrap_model(self, config: Config) -> None:
        self._model = DDP(self._model, **config)

    @property
    def module(self) -> nn.Module:
        return self._model.module

    if TYPE_CHECKING:

        @property
        def model(self) -> DDP:
            ...
