__all__ = [
    'FunctionalLoss',
    'NormMixin',
    'L1Loss',
    'MSELoss',
    'BCELoss',
    'BCEWithLogitsLoss',
    'CrossEntropyLoss',
    'CosineEmbeddingLoss',
]

from abc import ABC
from typing import Callable

import torch
import torch.nn.functional as F

from ...bases.configs import Config
from ...bases.registries import Item, RegistryMeta
from ...registries import ModelRegistry
from ..registries import LossRegistry
from .base import BaseLoss, Reduction


@LossRegistry.register_()
class FunctionalLoss(BaseLoss):

    def __init__(
        self,
        *args,
        func: Callable[..., torch.Tensor],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._func = func

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        if 'func' in config:
            config.func = ModelRegistry.build_or_return(config.func)
        return config

    def forward(  # pylint: disable=arguments-differ
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *args,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if mask is None:
            return self._func(
                pred,
                target,
                *args,
                reduction=self.reduction,
                **kwargs,
            )
        loss = self._func(
            pred,
            target,
            *args,
            reduction=Reduction.NONE.value,
            **kwargs,
        )
        return self._reduce(loss, mask)


class NormMixin(FunctionalLoss, ABC):

    def __init__(self, *args, norm: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._norm = norm

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *args,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self._norm:
            pred = F.normalize(pred)
            target = F.normalize(target)
        return super().forward(pred, target, *args, mask=mask, **kwargs)


@LossRegistry.register_()
class L1Loss(NormMixin, FunctionalLoss):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, func=F.l1_loss, **kwargs)


@LossRegistry.register_()
class MSELoss(NormMixin, FunctionalLoss):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, func=F.mse_loss, **kwargs)


@LossRegistry.register_()
class BCELoss(FunctionalLoss):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, func=F.binary_cross_entropy, **kwargs)


@LossRegistry.register_()
class BCEWithLogitsLoss(FunctionalLoss):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            func=F.binary_cross_entropy_with_logits,
            **kwargs,
        )


@LossRegistry.register_()
class CrossEntropyLoss(FunctionalLoss):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, func=F.cross_entropy, **kwargs)


@LossRegistry.register_()
class CosineEmbeddingLoss(FunctionalLoss):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, func=F.cosine_embedding_loss, **kwargs)
