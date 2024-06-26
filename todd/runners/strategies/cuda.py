__all__ = [
    'CUDAStrategy',
]

from typing import TypeVar

import torch
import torch.distributed as dist
from torch import nn

from ...bases.configs import Config
from ...patches.torch import get_local_rank
from ...utils import Store
from ..registries import StrategyRegistry
from .base import BaseStrategy

T = TypeVar('T', bound=nn.Module)


@StrategyRegistry.register_()
class CUDAStrategy(BaseStrategy[T]):

    def setup(self, config: Config) -> None:
        assert Store.cuda
        if not dist.is_initialized():
            init_process_group = config.get(
                'init_process_group',
                Config(backend='nccl'),
            )
            dist.init_process_group(**init_process_group)
        torch.cuda.set_device(get_local_rank() % torch.cuda.device_count())

    def map_model(self, model: nn.Module, config: Config) -> nn.Module:
        model = super().map_model(model, config)
        return model.cuda()
