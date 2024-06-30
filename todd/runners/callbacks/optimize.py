__all__ = [
    'OptimizeCallback',
]

from typing import Any, Mapping, TypeVar

import torch
from torch import nn

from ...bases.configs import Config
from ...bases.registries import BuildPreHookMixin, Item, RegistryMeta
from ...registries import ClipGradRegistry
from ..memo import Memo
from ..registries import CallbackRegistry
from .base import BaseCallback

T = TypeVar('T', bound=nn.Module)


@CallbackRegistry.register_()
class OptimizeCallback(BuildPreHookMixin, BaseCallback[T]):

    # TODO: add accumulate
    def __init__(
        self,
        *args,
        grad_scaler: torch.cuda.amp.GradScaler | None = None,
        grad_clipper: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if grad_scaler is not None:
            self._grad_scaler = grad_scaler
        if grad_clipper is not None:
            self._grad_clipper = grad_clipper

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        if isinstance(grad_scaler := config.get('grad_scaler'), Config):
            config.grad_scaler = torch.cuda.amp.GradScaler(**grad_scaler)
        if 'grad_clipper' in config:
            config.grad_clipper = ClipGradRegistry.build_or_return(
                config.grad_clipper,
            )
        return config

    @property
    def with_grad_scaler(self) -> bool:
        return hasattr(self, '_grad_scaler')

    @property
    def with_grad_clipper(self) -> bool:
        return hasattr(self, '_grad_clipper')

    def _scale_grad(self, loss: torch.Tensor) -> torch.Tensor:
        return self._grad_scaler.scale(loss)

    def _clip_grad(self, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        if self.with_grad_scaler:
            self._grad_scaler.unscale_(optimizer)
        parameters = [
            param for param_group in optimizer.param_groups
            for param in param_group['params']
        ]
        return self._grad_clipper(parameters)

    def _step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.with_grad_scaler:
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
        else:
            optimizer.step()

    def after_run_iter(self, batch, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        log: dict[str, Any] | None = memo.get('log')

        optimizer = self.trainer.optimizer
        loss: torch.Tensor = memo['loss']
        if self.with_grad_scaler:
            loss = self._scale_grad(loss)
        loss.backward()
        if self.with_grad_clipper:
            grad = self._clip_grad(optimizer)
            if log is not None:
                log['grad'] = f'{grad:.3f}'
        self._step(optimizer)
        optimizer.zero_grad()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)
        if self.with_grad_scaler:
            self._grad_scaler.load_state_dict(state_dict['grad_scaler'])

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        if self.with_grad_scaler:
            state_dict['grad_scaler'] = self._grad_scaler.state_dict()
        return state_dict
