__all__ = [
    'ComposedCallback',
]

from collections import UserList
from typing import Iterable

from ...base import CallbackRegistry, Config
from .base import BaseCallback


@CallbackRegistry.register()
class ComposedCallback(BaseCallback, UserList[BaseCallback]):

    def __init__(self, callbacks: Iterable[Config]) -> None:
        super().__init__(CallbackRegistry.build(c) for c in callbacks)

    def should_break(self, *args, **kwargs) -> bool:
        return any(c.should_break(*args, **kwargs) for c in self)

    def should_continue(self, *args, **kwargs) -> bool:
        return any(c.should_continue(*args, **kwargs) for c in self)

    def before_run_iter(self, *args, **kwargs) -> None:
        for c in self:
            c.before_run_iter(*args, **kwargs)

    def after_run_iter(self, *args, **kwargs) -> None:
        for c in self:
            c.after_run_iter(*args, **kwargs)

    def should_break_epoch(self, *args, **kwargs) -> bool:
        return any(c.should_break_epoch(*args, **kwargs) for c in self)

    def should_continue_epoch(self, *args, **kwargs) -> bool:
        return any(c.should_continue_epoch(*args, **kwargs) for c in self)

    def before_run_epoch(self, *args, **kwargs) -> None:
        for c in self:
            c.before_run_epoch(*args, **kwargs)

    def after_run_epoch(self, *args, **kwargs) -> None:
        for c in self:
            c.after_run_epoch(*args, **kwargs)

    def before_run(self, *args, **kwargs) -> None:
        for c in self:
            c.before_run(*args, **kwargs)

    def after_run(self, *args, **kwargs) -> None:
        for c in self:
            c.after_run(*args, **kwargs)
