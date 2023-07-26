__all__ = [
    'RunnerHolderMixin',
]

import weakref
from typing import cast

from .base import BaseRunner
from .epoch_based_trainer import EpochBasedTrainer
from .iter_based_trainer import IterBasedTrainer
from .trainer import Trainer
from .validator import Validator


class RunnerHolderMixin:

    def __init__(self, *args, runner: BaseRunner, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        runner_proxy = (
            runner if isinstance(runner, weakref.ProxyTypes) else
            weakref.proxy(runner)
        )
        self._runner = cast(BaseRunner, runner_proxy)

    @property
    def trainer(self) -> Trainer:
        assert isinstance(self._runner, Trainer)
        return self._runner

    @property
    def validator(self) -> Validator:
        assert isinstance(self._runner, Validator)
        return self._runner

    @property
    def runner(self) -> BaseRunner:
        return self._runner

    @property
    def iter_based_trainer(self) -> IterBasedTrainer:
        assert isinstance(self._runner, IterBasedTrainer)
        return self._runner

    @property
    def epoch_based_trainer(self) -> EpochBasedTrainer:
        assert isinstance(self._runner, EpochBasedTrainer)
        return self._runner
