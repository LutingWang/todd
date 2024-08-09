__all__ = [
    'AutocastCallback',
]

import contextlib
from typing import Any, TypeVar

import torch
from torch import nn

from ...bases.configs import Config
from ..memo import Memo
from ..registries import CallbackRegistry
from .base import BaseCallback

T = TypeVar('T', bound=nn.Module)


@CallbackRegistry.register_()
class AutocastCallback(BaseCallback[T]):

    def __init__(self, *args, autocast: Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._autocast = autocast

    def run_iter_context(
        self,
        exit_stack: contextlib.ExitStack,
        batch: Any,
        memo: Memo,
    ) -> None:
        super().run_iter_context(exit_stack, batch, memo)
        exit_stack.enter_context(torch.autocast(**self._autocast))
