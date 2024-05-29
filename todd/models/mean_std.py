__all__ = [
    'MeanStdMixin',
]

from abc import ABC

import einops
import torch
from torch import nn


class MeanStdMixin(nn.Module, ABC):

    def __init__(
        self,
        *args,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        mean_ = torch.tensor(mean)
        mean_ = einops.rearrange(mean_, 'c -> 1 c 1 1')
        std_ = torch.tensor(std)
        std_ = einops.rearrange(std_, 'c -> 1 c 1 1')
        self.register_buffer('_mean', mean_)
        self.register_buffer('_std', std_)

    @property
    def mean(self) -> torch.Tensor:
        return self.get_buffer('_mean')

    @property
    def std(self) -> torch.Tensor:
        return self.get_buffer('_std')

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        return image

    def denormalize(self, image: torch.Tensor) -> torch.Tensor:
        image = image * self.std + self.mean
        return image
