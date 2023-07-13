__all__ = [
    'ComposedCallback',
]

import contextlib
from collections import UserList
from typing import Any, Iterable, Literal, Mapping, TypedDict

from ...base import CallbackRegistry, Config
from ..runners import BaseRunner, EpochBasedTrainer
from .base import BaseCallback

Memo = dict[str, Any]


class Priority(TypedDict, total=False):
    connect: int
    should_break: int
    should_continue: int
    before_run_iter: int
    run_iter_context: int
    after_run_iter: int
    should_break_epoch: int
    should_continue_epoch: int
    before_run_epoch: int
    run_epoch_context: int
    after_run_epoch: int
    before_run: int
    after_run: int


@CallbackRegistry.register()
class ComposedCallback(BaseCallback, UserList[BaseCallback]):

    def __init__(self, callbacks: Iterable[Config]) -> None:
        self._priorities: list[Priority] = [
            c.pop('priority', dict()) for c in callbacks
        ]
        super().__init__(CallbackRegistry.build(c) for c in callbacks)

    def _callbacks(
        self,
        name: Literal['connect', 'should_break', 'should_continue',
                      'before_run_iter', 'run_iter_context', 'after_run_iter',
                      'should_break_epoch', 'should_continue_epoch',
                      'before_run_epoch', 'run_epoch_context',
                      'after_run_epoch', 'before_run', 'after_run'],
    ) -> list[BaseCallback]:
        priorities = [p.get(name, 0) for p in self._priorities]
        priority_index = [(p, i) for i, p in enumerate(priorities)]
        priority_index = sorted(priority_index)
        _, indices = zip(*priority_index)
        return [self[i] for i in indices]

    def connect(self, runner: BaseRunner) -> None:
        super().connect(runner)
        for c in self._callbacks('connect'):
            c.connect(runner)

    def should_break(self, *args, **kwargs) -> bool:
        return any(
            c.should_break(*args, **kwargs)
            for c in self._callbacks('should_break')
        )

    def should_continue(self, *args, **kwargs) -> bool:
        return any(
            c.should_continue(*args, **kwargs)
            for c in self._callbacks('should_continue')
        )

    def before_run_iter(self, *args, **kwargs) -> None:
        for c in self._callbacks('before_run_iter'):
            c.before_run_iter(*args, **kwargs)

    def run_iter_context(
        self,
        runner: BaseRunner,
        exit_stack: contextlib.ExitStack,
        batch,
        memo: Memo,
    ) -> None:
        super().run_iter_context(runner, exit_stack, batch, memo)
        for c in self._callbacks('run_iter_context'):
            c.run_iter_context(runner, exit_stack, batch, memo)

    def after_run_iter(self, *args, **kwargs) -> None:
        for c in self._callbacks('after_run_iter'):
            c.after_run_iter(*args, **kwargs)

    def should_break_epoch(self, *args, **kwargs) -> bool:
        return any(
            c.should_break_epoch(*args, **kwargs)
            for c in self._callbacks('should_break_epoch')
        )

    def should_continue_epoch(self, *args, **kwargs) -> bool:
        return any(
            c.should_continue_epoch(*args, **kwargs)
            for c in self._callbacks('should_continue_epoch')
        )

    def before_run_epoch(self, *args, **kwargs) -> None:
        for c in self._callbacks('before_run_epoch'):
            c.before_run_epoch(*args, **kwargs)

    def run_epoch_context(
        self,
        runner: EpochBasedTrainer,
        exit_stack: contextlib.ExitStack,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        super().run_epoch_context(runner, exit_stack, epoch_memo, memo)
        for c in self._callbacks('run_epoch_context'):
            c.run_epoch_context(runner, exit_stack, epoch_memo, memo)

    def after_run_epoch(self, *args, **kwargs) -> None:
        for c in self._callbacks('after_run_epoch'):
            c.after_run_epoch(*args, **kwargs)

    def before_run(self, *args, **kwargs) -> None:
        for c in self._callbacks('before_run'):
            c.before_run(*args, **kwargs)

    def after_run(self, *args, **kwargs) -> None:
        for c in self._callbacks('after_run'):
            c.after_run(*args, **kwargs)

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['callbacks'] = [c.state_dict(*args, **kwargs) for c in self]
        return state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)
        for c, s in zip(self, state_dict['callbacks'], strict=True):
            c.load_state_dict(s, *args, **kwargs)
