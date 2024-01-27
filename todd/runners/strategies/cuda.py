__all__ = [
    'CUDAStrategy',
]

from typing import TypeVar

import torch
import torch.distributed
from torch import nn

from ...base import Config, Store, StrategyRegistry
from ...utils import get_local_rank
from .base import BaseStrategy

T = TypeVar('T', bound=nn.Module)


@StrategyRegistry.register_()
class CUDAStrategy(BaseStrategy[T]):

    def __init__(
        self,
        *args,
        setup: Config | None = None,
        **kwargs,
    ) -> None:
        assert Store.CUDA
        if setup is None:
            setup = Config()
        self._setup(setup)
        super().__init__(*args, **kwargs)

    def _setup(self, config: Config) -> None:
        if not torch.distributed.is_initialized():
            init_process_group = config.get(
                'init_process_group',
                Config(backend='nccl'),
            )
            torch.distributed.init_process_group(**init_process_group)
        torch.cuda.set_device(get_local_rank() % torch.cuda.device_count())

    def map_model(
        self,
        model: nn.Module,
        config: Config | None = None,
    ) -> nn.Module:
        model = super().map_model(model, config)
        return model.cuda()
