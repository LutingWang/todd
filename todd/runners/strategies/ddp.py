__all__ = [
    'DDPStrategy',
]

from typing import TYPE_CHECKING, Any, Mapping

import torch
import torch.distributed
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ...base import Config, OptimizerRegistry, Store, StrategyRegistry
from ...utils import get_local_rank
from .base import BaseStrategy


@StrategyRegistry.register()
class DDPStrategy(BaseStrategy):
    _model: DDP

    def __init__(
        self,
        *args,
        setup: Config | None = None,
        wrap_model: Config | None = None,
        **kwargs,
    ) -> None:
        assert Store.CUDA
        if setup is None:
            setup = Config()
        self._setup(setup)
        super().__init__(*args, **kwargs)
        if wrap_model is None:
            wrap_model = Config()
        self._wrap_model(wrap_model)

    def _setup(self, config: Config) -> None:
        init_process_group = config.get(
            'init_process_group',
            Config(backend='nccl'),
        )
        torch.distributed.init_process_group(**init_process_group)
        torch.cuda.set_device(get_local_rank() % torch.cuda.device_count())

    def _wrap_model(self, config: Config) -> None:
        self._model = DDP(self._model.cuda(), **config)

    def build_optimizer(self, config: Config) -> torch.optim.Optimizer:
        return OptimizerRegistry.build(config, model=self._model.module)

    def model_state_dict(self, *args, **kwargs) -> dict[str, Any]:
        return self.module.state_dict(*args, **kwargs)

    def load_model_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        self.module.load_state_dict(state_dict, *args, **kwargs)

    @property
    def module(self) -> nn.Module:
        return self._model.module

    if TYPE_CHECKING:

        @property
        def model(self) -> DDP:
            ...
