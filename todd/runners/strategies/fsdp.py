from typing import TYPE_CHECKING, Any, Mapping

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...base import Config, StrategyRegistry
from ..runners import Trainer
from .ddp import DDPStrategy

# TODO: update when pytorch updates


@StrategyRegistry.register()
class FSDPStrategy(DDPStrategy):
    _model: FSDP

    def _wrap_model(self, config: Config) -> None:
        self._model = FSDP(
            self._model.cuda(),
            **config,
        )

    def optim_state_dict(
        self,
        runner: Trainer,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        return FSDP.full_optim_state_dict(self._model, runner.optimizer)

    def load_optim_state_dict(
        self,
        runner: Trainer,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        sharded_state_dict = FSDP.scatter_full_optim_state_dict(
            state_dict,
            self._model,
        )
        runner.optimizer.load_state_dict(sharded_state_dict)

    if TYPE_CHECKING:

        @property
        def model(self) -> FSDP:
            ...
