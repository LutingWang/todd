__all__ = [
    'ComposedCallback',
]

from typing import Any, Iterable, Iterator, Literal, Mapping, TypeVar

from torch import nn

from ...bases.configs import Config
from ...bases.registries import BuildPreHookMixin, Item, RegistryMeta
from ..registries import CallbackRegistry
from ..utils import PriorityQueue
from .base import BaseCallback
from .log import LogCallback
from .lr import LRScheduleCallback
from .optimize import OptimizeCallback

T = TypeVar('T', bound=nn.Module)
KT = Literal['bind', 'should_break', 'should_continue', 'before_run_iter',
             'run_iter_context', 'after_run_iter', 'should_break_epoch',
             'should_continue_epoch', 'before_run_epoch', 'run_epoch_context',
             'after_run_epoch', 'before_run', 'after_run']
Priority = Mapping[KT, int]


@CallbackRegistry.register_()
class ComposedCallback(BuildPreHookMixin, BaseCallback[T]):

    def __init__(
        self,
        *args,
        priorities: Iterable[Priority],
        callbacks: Iterable[BaseCallback[T]],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._priority_queue = PriorityQueue(priorities, callbacks)
        self._check()

    def _check(self) -> None:
        assert not any(
            isinstance(callback, ComposedCallback)
            for callback in self._priority_queue.queue
        )

        callbacks = self._priority_queue('after_run_iter')
        optimize_indices = [
            i for i, callback in enumerate(callbacks)
            if isinstance(callback, OptimizeCallback)
        ]
        lr_schedule_indices = [
            i for i, callback in enumerate(callbacks)
            if isinstance(callback, LRScheduleCallback)
        ]
        log_indices = [
            i for i, callback in enumerate(callbacks)
            if isinstance(callback, LogCallback)
        ]
        assert (
            max(optimize_indices, default=-1)
            < min(lr_schedule_indices, default=len(callbacks))
        )
        assert (
            max(optimize_indices, default=-1)
            < min(log_indices, default=len(callbacks))
        )
        assert (
            max(lr_schedule_indices, default=-1)
            < min(log_indices, default=len(callbacks))
        )

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        callbacks: Iterable[Config] = config.callbacks
        config.priorities = [c.pop('priority', dict()) for c in callbacks]
        config.callbacks = [registry.build_or_return(c) for c in callbacks]
        return config

    @property
    def callbacks(self) -> list[BaseCallback[T]]:
        return self._priority_queue.queue

    def put(
        self,
        callback: BaseCallback[T],
        priority: Priority | None = None,
    ) -> None:
        if priority is None:
            priority = dict()
        self._priority_queue.append((priority, callback))

    def __iter__(self) -> Iterator[BaseCallback[T]]:
        return iter(self._priority_queue.queue)

    def bind(self, *args, **kwargs) -> None:
        super().bind(*args, **kwargs)
        for c in self._priority_queue('bind'):
            c.bind(*args, **kwargs)

    def should_break(self, *args, **kwargs) -> bool:
        return super().should_break(*args, **kwargs) or any(
            c.should_break(*args, **kwargs)
            for c in self._priority_queue('should_break')
        )

    def should_continue(self, *args, **kwargs) -> bool:
        return super().should_continue(*args, **kwargs) or any(
            c.should_continue(*args, **kwargs)
            for c in self._priority_queue('should_continue')
        )

    def before_run_iter(self, *args, **kwargs) -> None:
        super().before_run_iter(*args, **kwargs)
        for c in self._priority_queue('before_run_iter'):
            c.before_run_iter(*args, **kwargs)

    def run_iter_context(self, *args, **kwargs) -> None:
        super().run_iter_context(*args, **kwargs)
        for c in self._priority_queue('run_iter_context'):
            c.run_iter_context(*args, **kwargs)

    def after_run_iter(self, *args, **kwargs) -> None:
        super().after_run_iter(*args, **kwargs)
        for c in self._priority_queue('after_run_iter'):
            c.after_run_iter(*args, **kwargs)

    def should_break_epoch(self, *args, **kwargs) -> bool:
        return super().should_break_epoch(*args, **kwargs) or any(
            c.should_break_epoch(*args, **kwargs)
            for c in self._priority_queue('should_break_epoch')
        )

    def should_continue_epoch(self, *args, **kwargs) -> bool:
        return super().should_continue_epoch(*args, **kwargs) or any(
            c.should_continue_epoch(*args, **kwargs)
            for c in self._priority_queue('should_continue_epoch')
        )

    def before_run_epoch(self, *args, **kwargs) -> None:
        super().before_run_epoch(*args, **kwargs)
        for c in self._priority_queue('before_run_epoch'):
            c.before_run_epoch(*args, **kwargs)

    def run_epoch_context(self, *args, **kwargs) -> None:
        super().run_epoch_context(*args, **kwargs)
        for c in self._priority_queue('run_epoch_context'):
            c.run_epoch_context(*args, **kwargs)

    def after_run_epoch(self, *args, **kwargs) -> None:
        super().after_run_epoch(*args, **kwargs)
        for c in self._priority_queue('after_run_epoch'):
            c.after_run_epoch(*args, **kwargs)

    def before_run(self, *args, **kwargs) -> None:
        super().before_run(*args, **kwargs)
        for c in self._priority_queue('before_run'):
            c.before_run(*args, **kwargs)

    def after_run(self, *args, **kwargs) -> None:
        super().after_run(*args, **kwargs)
        for c in self._priority_queue('after_run'):
            c.after_run(*args, **kwargs)

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['callbacks'] = [
            c.state_dict(*args, **kwargs) for c in self._priority_queue.queue
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
            self._priority_queue.queue,
            state_dict['callbacks'],
            strict=True,
        ):
            c.load_state_dict(s, *args, **kwargs)
