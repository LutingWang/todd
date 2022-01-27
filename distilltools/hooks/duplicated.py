from typing import Dict, List

import torch

from .builder import HOOKS
from .standard import StandardHook


@HOOKS.register_module()
class DuplicatedHook(StandardHook):
    def __init__(self, *args, num: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._num = num

    @property
    def tensor(self) -> Dict[str, List[torch.Tensor]]:
        return {self.id_: [self._tensor] * self._num}
