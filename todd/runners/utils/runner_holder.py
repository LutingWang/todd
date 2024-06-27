__all__ = [
    'RunnerHolderMixin',
]

from typing import TypeVar

from torch import nn

from ...utils import HolderMixin
from ..base import BaseRunner
from ..epoch_based_trainer import EpochBasedTrainer
from ..iter_based_trainer import IterBasedTrainer
from ..trainer import Trainer
from ..validator import Validator

T = TypeVar('T', bound=nn.Module)


class RunnerHolderMixin(HolderMixin[BaseRunner[T]]):

    def __init__(
        self,
        *args,
        runner: BaseRunner[T] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, instance=runner, **kwargs)

    @property
    def runner(self) -> BaseRunner[T]:
        return self._instance

    @property
    def trainer(self) -> Trainer[T]:
        assert isinstance(self._instance, Trainer)
        return self._instance

    @property
    def validator(self) -> Validator[T]:
        assert isinstance(self._instance, Validator)
        return self._instance

    @property
    def iter_based_trainer(self) -> IterBasedTrainer[T]:
        assert isinstance(self._instance, IterBasedTrainer)
        return self._instance

    @property
    def epoch_based_trainer(self) -> EpochBasedTrainer[T]:
        assert isinstance(self._instance, EpochBasedTrainer)
        return self._instance
