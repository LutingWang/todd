__all__ = [
    'BaseShadow',
]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from ...data_structures import TreeUtil
from ...utils import StateDict
from ..norms import BATCHNORMS


class BaseShadow(nn.Module, ABC):

    def __init__(
        self,
        *args,
        module: nn.Module,
        device=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # BN layers are not supported
        assert not any(isinstance(m, BATCHNORMS) for m in module.modules())

        self._shadow = self._to_device(module.state_dict())
        self._device = device

    @property
    def shadow(self) -> StateDict:
        return self._shadow

    @shadow.setter
    def shadow(self, shadow: StateDict) -> None:
        self._shadow = self._to_device(shadow)

    def _to_device(self, state_dict: StateDict) -> StateDict:
        if self._device is None:
            return state_dict
        return TreeUtil.map(
            lambda t: cast(torch.Tensor, t).to(self._device),
            state_dict,
        )

    @abstractmethod
    def _forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, module: nn.Module) -> None:
        for k, v in self._to_device(module.state_dict()).items():
            self._shadow[k] = self._forward(self._shadow[k], v)

    if TYPE_CHECKING:
        __call__ = forward
