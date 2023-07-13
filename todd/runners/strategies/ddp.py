__all__ = [
    'DDPStrategy',
]

import torch
import torch.distributed
import torch.nn as nn

from ...base import Config, Store, StrategyRegistry
from ...utils import get_local_rank
from .base import BaseStrategy


@StrategyRegistry.register()
class DDPStrategy(BaseStrategy):

    def __init__(
        self,
        *args,
        init_process_group: Config | None = None,
        ddp: Config | None = None,
        **kwargs,
    ) -> None:
        assert Store.CUDA
        super().__init__(*args, **kwargs)

        if init_process_group is None:
            init_process_group = Config(backend='nccl')
        torch.distributed.init_process_group(**init_process_group)
        torch.cuda.set_device(get_local_rank() % torch.cuda.device_count())

        if ddp is None:
            ddp = Config()
        self._model = nn.parallel.DistributedDataParallel(
            self._model.cuda(),
            **ddp,
        )

    @property
    def model(self) -> nn.Module:
        return self._model.module
