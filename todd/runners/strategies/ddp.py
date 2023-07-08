__all__ = [
    'DDPStrategy',
]

from typing import Any

import torch
import torch.distributed
import torch.nn as nn

from ...base import Config, StrategyRegistry
from ...base.patches import get_local_rank
from ...base.stores import Store
from .base import BaseStrategy


@StrategyRegistry.register()
class DDPStrategy(BaseStrategy):

    def __init__(
        self,
        *args,
        setup: Config | None = None,
        wrap_model: Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._setup = Config(backend='nccl') if setup is None else setup
        self._wrap_model = Config() if wrap_model is None else wrap_model

    def setup(self) -> None:
        assert Store.CUDA
        torch.distributed.init_process_group(**self._setup)
        torch.cuda.set_device(get_local_rank() % torch.cuda.device_count())

    def wrap_model(
        self,
        model: nn.Module,
    ) -> nn.parallel.DistributedDataParallel:
        return nn.parallel.DistributedDataParallel(
            model.cuda(),
            **self._wrap_model,
        )

    def get_model(
        self,
        model: nn.parallel.DistributedDataParallel,
    ) -> nn.Module:
        return model.module

    def state_dict(
        self,
        model: nn.parallel.DistributedDataParallel,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        module: nn.Module = model.module
        return module.state_dict(*args, **kwargs)

    def load_state_dict(
        self,
        model: nn.parallel.DistributedDataParallel,
        state_dict: dict[str, Any],
        *args,
        **kwargs,
    ) -> None:
        module: nn.Module = model.module
        module.load_state_dict(state_dict, *args, **kwargs)
