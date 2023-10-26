__all__ = [
    'Validator',
]

import torch
import torch.distributed
import torch.utils.data

from .base import BaseRunner, RunnerRegistry
from .types import Memo


@RunnerRegistry.register_()
class Validator(BaseRunner):

    def _setup(self) -> Memo:
        self._strategy.model.eval()
        return super()._setup()

    @torch.no_grad()
    def run(self) -> Memo:
        return super().run()
