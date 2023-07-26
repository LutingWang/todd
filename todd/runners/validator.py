__all__ = [
    'Validator',
]

from typing import Any

import torch
import torch.distributed
import torch.utils.data

from .base import BaseRunner

Memo = dict[str, Any]


class Validator(BaseRunner):

    def _setup(self) -> Memo:
        self._strategy.model.eval()
        return super()._setup()

    @torch.no_grad()
    def run(self) -> Memo:
        return super().run()
