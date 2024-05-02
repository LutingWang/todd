__all__ = [
    'AutocastCallback',
]

import contextlib

import torch

from ...base import CallbackRegistry
from ...configs import Config
from ..types import Memo
from .base import BaseCallback


@CallbackRegistry.register_()
class AutocastCallback(BaseCallback):

    def __init__(self, *args, autocast: Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._autocast = autocast

    def run_iter_context(
        self,
        exit_stack: contextlib.ExitStack,
        batch,
        memo: Memo,
    ) -> None:
        exit_stack.enter_context(torch.autocast(**self._autocast))
