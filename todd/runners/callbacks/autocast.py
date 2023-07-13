__all__ = [
    'AutocastCallback',
]

import contextlib
from typing import Any

import torch

from ...base import CallbackRegistry, Config
from ..runners import BaseRunner
from .base import BaseCallback

Memo = dict[str, Any]


@CallbackRegistry.register()
class AutocastCallback(BaseCallback):

    def __init__(self, *args, autocast: Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._autocast = autocast

    def run_iter_context(
        self,
        runner: BaseRunner,
        exit_stack: contextlib.ExitStack,
        batch,
        memo: Memo,
    ) -> None:
        exit_stack.enter_context(torch.autocast(**self._autocast))
