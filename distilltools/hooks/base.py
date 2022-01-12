from abc import abstractmethod, abstractproperty
from typing import Any, Dict, Optional

import torch.nn as nn

from ..utils import getattr_recur


class BaseHook:
    def __init__(self, id_: str, alias: Optional[str] = None, on_input: bool = False):
        self._id = id_
        self._alias = alias
        self._on_input = on_input
        self.reset()

    @property
    def id_(self) -> str:
        return self._id

    @property
    def alias(self) -> str:
        return self._alias or self._id

    @property
    def on_input(self) -> bool:
        return self._on_input

    @abstractproperty
    def tensor(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def register_tensor(self, _: Any):
        pass

    def _forward_hook(self, module: nn.Module, input_: Any, output: Any):
        self.register_tensor(input_ if self._on_input else output)

    def register_hook(self, model: nn.Module):
        module: nn.Module = getattr_recur(model, self.id_)
        module.register_forward_hook(self._forward_hook)
