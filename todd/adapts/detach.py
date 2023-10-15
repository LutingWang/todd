import torch

from .base import AdaptRegistry, BaseAdapt


@AdaptRegistry.register_()
class Detach(BaseAdapt):

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach()


@AdaptRegistry.register_()
class ListDetach(BaseAdapt):

    def forward(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        return [tensor.detach() for tensor in tensors]
