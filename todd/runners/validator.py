__all__ = [
    'Validator',
]

from typing import TypeVar

import torch
from torch import nn

from .base import BaseRunner, RunnerRegistry
from .types import Memo

T = TypeVar('T', bound=nn.Module)


@RunnerRegistry.register_()
class Validator(BaseRunner[T]):

    def _setup(self) -> Memo:
        self._model.eval()
        return super()._setup()

    @torch.no_grad()
    def run(self) -> Memo:
        return super().run()
