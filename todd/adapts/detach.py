from typing import List

import torch

from .base import ADAPTS, BaseAdapt


@ADAPTS.register_module()
class Detach(BaseAdapt):

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach()


@ADAPTS.register_module()
class ListDetach(BaseAdapt):

    def forward(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        return [tensor.detach() for tensor in tensors]
