__all__ = [
    'BaseShadow',
]

from abc import ABC, abstractmethod
from functools import partial
from typing import Any

import torch
from torch import nn

from ...utils import NestedTensorCollectionUtils, StateDict
from ..norms import BATCHNORMS


class BaseShadow(nn.Module, ABC):

    def __init__(
        self,
        *args,
        module: nn.Module,
        device: Any = None,
        **kwargs,
    ) -> None:
        # BN layers are not supported
        assert not any(isinstance(m, BATCHNORMS) for m in module.modules())

        super().__init__(*args, **kwargs)
        self._device = device
        self._shadow = self._state_dict_to_device(module)

    @property
    def shadow(self) -> StateDict:
        return self._shadow

    @shadow.setter
    def shadow(self, value: StateDict) -> None:
        self._shadow = self._to_device(value)

    def _to_device(self, state_dict: StateDict) -> StateDict:
        if self._device is None:
            return state_dict
        utils = NestedTensorCollectionUtils()
        f = partial(torch.Tensor.to, device=self._device)
        return utils.map(f, state_dict)

    def _state_dict_to_device(self, module: nn.Module) -> StateDict:
        return self._to_device(module.state_dict())

    @abstractmethod
    def _forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, module: nn.Module) -> None:
        utils = NestedTensorCollectionUtils()
        self._shadow = utils.map(
            self._forward,  # type: ignore[arg-type]
            self._shadow,
            self._state_dict_to_device(module),
        )
