__all__ = [
    'Validator',
]

from typing import Any

import torch
import torch.distributed
import torch.utils.data

from .base import BaseRunner, RunnerRegistry

Memo = dict[str, Any]


@RunnerRegistry.register_()
class Validator(BaseRunner):

    def _setup(self) -> Memo:
        self._strategy.model.eval()
        return super()._setup()

    @torch.no_grad()
    def run(self) -> Memo:
        return super().run()
