__all__ = [
    'OptimizeCallback',
]

import contextlib
from typing import Any, Mapping, TypeVar

import torch
from torch import nn

from ...bases.configs import Config
from ...bases.registries import BuildPreHookMixin, Item, RegistryMeta
from ...patches.torch import (
    get_rank,
    named_trainable_parameters,
    named_training_modules,
)
from ...registries import ClipGradRegistry
from ..memo import Memo
from ..registries import CallbackRegistry
from .base import BaseCallback

T = TypeVar('T', bound=nn.Module)


@CallbackRegistry.register_()
class OptimizeCallback(BuildPreHookMixin, BaseCallback[T]):

    def __init__(
        self,
        *args,
        grad_scaler: torch.cuda.amp.GradScaler | None = None,
        grad_clipper: Any = None,
        accumulate: int = 1,
        check: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if grad_scaler is not None:
            self._grad_scaler = grad_scaler
        if grad_clipper is not None:
            self._grad_clipper = grad_clipper
        self._accumulate = accumulate
        self._check = check

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

    def _should_accumulate(self) -> bool:
        return self.trainer.iter_ % self._accumulate != 0

    def run_iter_context(
        self,
        exit_stack: contextlib.ExitStack,
        batch: Any,
        memo: Memo,
    ) -> None:
        super().run_iter_context(exit_stack, batch, memo)
        trainer = self.trainer
        if not self._should_accumulate():
            return
        no_sync = getattr(trainer.model, 'no_sync', None)
        if no_sync is not None:
            exit_stack.enter_context(no_sync())

    def before_run(self, memo: Memo) -> None:
        super().before_run(memo)
        if get_rank() > 0:
            return

        trainer = self.trainer
        logger = trainer.logger
        module = trainer.strategy.module

        training_modules = [
            repr(name) for name, _ in named_training_modules(module)
        ]
        logger.debug(
            "Training modules\n" + ", ".join(training_modules),
        )

        trainable_parameters = {
            repr(name): parameter.numel()
            for name, parameter in named_trainable_parameters(module)
        }
        logger.debug(
            "Requires grad parameters\n"
            + ", ".join(trainable_parameters.keys()),
        )

        num_trainable_parameters = sum(trainable_parameters.values())
        logger.debug(
            "Total number of requires grad parameters: %s",
            f'{num_trainable_parameters / 2**30:.3f} B'
            if num_trainable_parameters >= 2**30 else
            f'{num_trainable_parameters / 2**20:.3f} M',
        )

    def after_run_iter(self, batch: Any, memo: Memo) -> None:  # noqa: C901
        super().after_run_iter(batch, memo)
        log: dict[str, Any] | None = memo.get('log')

        trainer = self.trainer
        optimizer = trainer.optimizer
        module = trainer.strategy.module

        loss: torch.Tensor = memo['loss']

        if self.with_grad_scaler:
            loss = self._scale_grad(loss)

        loss.backward()

        if trainer.iter_ == 1 and self._check:
            for name, parameter in named_trainable_parameters(module):
                if parameter.grad is None:
                    trainer.logger.warning(
                        "Parameter %s received no gradient",
                        name,
                    )
                elif parameter.grad.isnan().any():
                    trainer.logger.warning(
                        "Parameter %s received NaN gradient",
                        name,
                    )

        if self.with_grad_clipper:
            grad = self._clip_grad(optimizer)
            if log is not None:
                log['grad'] = f'{grad:.3f}'

        if not self._should_accumulate():
            self._step(optimizer)
            optimizer.zero_grad()
            if trainer.iter_ == self._accumulate and self._check:
                for name, parameter in named_trainable_parameters(module):
                    if parameter.grad is not None:
                        trainer.logger.warning(
                            "Parameter %s gradient not cleared",
                            name,
                        )

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
