__all__ = [
    'ComposedCallback',
]

from typing import Any, Iterable, Iterator, Literal, Mapping

from ...configs import Config
from ..registries import CallbackRegistry
from ..utils import PriorityQueue
from .base import BaseCallback

KT = Literal['init', 'should_break', 'should_continue', 'before_run_iter',
             'run_iter_context', 'after_run_iter', 'should_break_epoch',
             'should_continue_epoch', 'before_run_epoch', 'run_epoch_context',
             'after_run_epoch', 'before_run', 'after_run']


@CallbackRegistry.register_()
class ComposedCallback(BaseCallback):

    def __init__(self, *args, callbacks: Iterable[Config], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        priorities = [c.pop('priority', dict()) for c in callbacks]
        queue = [
            CallbackRegistry.build(c, runner=self.runner) for c in callbacks
        ]
        self._callbacks: PriorityQueue[KT, BaseCallback] = \
            PriorityQueue(priorities, queue)

    @property
    def callbacks(self) -> PriorityQueue[KT, BaseCallback]:
        return self._callbacks

    def __iter__(self) -> Iterator[BaseCallback]:
        return iter(self._callbacks.queue)

    def init(self, *args, **kwargs) -> None:
        super().init(*args, **kwargs)
        for c in self._callbacks('init'):
            c.init(*args, **kwargs)

    def should_break(self, *args, **kwargs) -> bool:
        super().should_break(*args, **kwargs)
        return any(
            c.should_break(*args, **kwargs)
            for c in self._callbacks('should_break')
        )

    def should_continue(self, *args, **kwargs) -> bool:
        super().should_continue(*args, **kwargs)
        return any(
            c.should_continue(*args, **kwargs)
            for c in self._callbacks('should_continue')
        )

    def before_run_iter(self, *args, **kwargs) -> None:
        super().before_run_iter(*args, **kwargs)
        for c in self._callbacks('before_run_iter'):
            c.before_run_iter(*args, **kwargs)

    def run_iter_context(self, *args, **kwargs) -> None:
        super().run_iter_context(*args, **kwargs)
        for c in self._callbacks('run_iter_context'):
            c.run_iter_context(*args, **kwargs)

    def after_run_iter(self, *args, **kwargs) -> None:
        super().after_run_iter(*args, **kwargs)
        for c in self._callbacks('after_run_iter'):
            c.after_run_iter(*args, **kwargs)

    def should_break_epoch(self, *args, **kwargs) -> bool:
        super().should_break_epoch(*args, **kwargs)
        return any(
            c.should_break_epoch(*args, **kwargs)
            for c in self._callbacks('should_break_epoch')
        )

    def should_continue_epoch(self, *args, **kwargs) -> bool:
        super().should_continue_epoch(*args, **kwargs)
        return any(
            c.should_continue_epoch(*args, **kwargs)
            for c in self._callbacks('should_continue_epoch')
        )

    def before_run_epoch(self, *args, **kwargs) -> None:
        super().before_run_epoch(*args, **kwargs)
        for c in self._callbacks('before_run_epoch'):
            c.before_run_epoch(*args, **kwargs)

    def run_epoch_context(self, *args, **kwargs) -> None:
        super().run_epoch_context(*args, **kwargs)
        for c in self._callbacks('run_epoch_context'):
            c.run_epoch_context(*args, **kwargs)

    def after_run_epoch(self, *args, **kwargs) -> None:
        super().after_run_epoch(*args, **kwargs)
        for c in self._callbacks('after_run_epoch'):
            c.after_run_epoch(*args, **kwargs)

    def before_run(self, *args, **kwargs) -> None:
        super().before_run(*args, **kwargs)
        for c in self._callbacks('before_run'):
            c.before_run(*args, **kwargs)

    def after_run(self, *args, **kwargs) -> None:
        super().after_run(*args, **kwargs)
        for c in self._callbacks('after_run'):
            c.after_run(*args, **kwargs)

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['callbacks'] = [
            c.state_dict(*args, **kwargs) for c in self._callbacks.queue
        ]
        return state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)
        for c, s in zip(
            self._callbacks.queue,
            state_dict['callbacks'],
            strict=True,
        ):
            c.load_state_dict(s, *args, **kwargs)
