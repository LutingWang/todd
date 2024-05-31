__all__ = [
    'Validator',
]

from typing import TypeVar

import torch
from torch import nn

from ..registries import RunnerRegistry
from .base import BaseRunner
from .memo import Memo

T = TypeVar('T', bound=nn.Module)


@RunnerRegistry.register_()
class Validator(BaseRunner[T]):

    @property
    def iters(self) -> int:
        return len(self._dataloader)

    def _setup(self) -> Memo:
        self._model.eval()
        memo = super()._setup()
        memo['dataloader'] = self._dataloader
        return memo

    @torch.no_grad()
    def run(self) -> Memo:
        return super().run()
