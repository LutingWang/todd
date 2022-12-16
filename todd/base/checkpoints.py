__all__ = [
    'load_open_mmlab_models',
    'transfer_weight',
    'transfer_weights',
    'save_checkpoint',
    'load_checkpoint',
]

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ._extensions import Module, getattr_recur
from .configs import Config
from .loggers import get_logger
from .registries import RegistryMeta


def load_open_mmlab_models(
    registry: RegistryMeta,
    config: Config,
    config_options: Config | None = None,  # TODO: rename overload
    ckpt: str | None = None,
) -> Module:
    model = (
        registry.build(config)
        if config_options is None else registry.build(config_options, config)
    )
    if ckpt is not None:
        import mmcv.runner
        mmcv.runner.load_checkpoint(model, ckpt, map_location='cpu')
        model._is_init = True
    return model


def transfer_weight(target: Module, source: Module) -> None:
    state_dict = source.state_dict()
    missing_keys, unexpected_keys = target.load_state_dict(
        state_dict,
        strict=False,
    )
    if missing_keys:
        get_logger().warning('missing_keys:', missing_keys)
    if unexpected_keys:
        get_logger().warning('unexpected_keys:', unexpected_keys)
    target._is_init = True  # type: ignore[assignment]


def transfer_weights(models, weight_prefixes: dict[str, str]) -> None:
    for target_prefix, source_prefix in weight_prefixes.items():
        target = getattr_recur(models, target_prefix)
        source = getattr_recur(models, source_prefix)
        transfer_weight(target, source)


def save_checkpoint(
    model: nn.Module,
    f: str,
    *,
    optimizer: optim.Optimizer | None = None,
    scheduler: lr_scheduler._LRScheduler | None = None,
    meta: Config | None = None,
    **kwargs,
) -> None:
    checkpoint: dict[str, Any] = dict()
    checkpoint['model'] = model.state_dict(
        **kwargs.get('model_config', dict()),
    )
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict(
            **kwargs.get('optimizer_config', dict()),
        )
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict(
            **kwargs.get('scheduler_config', dict()),
        )
    if meta is not None:
        checkpoint['meta'] = meta
    torch.save(checkpoint, f, **kwargs.get('save_config', dict()))


def load_checkpoint(
    model: nn.Module,
    f: str,
    *,
    optimizer: optim.Optimizer | None = None,
    scheduler: lr_scheduler._LRScheduler | None = None,
    **kwargs,
) -> Config | None:
    checkpoint: dict[str, Any] = torch.load(
        f,
        **kwargs.get('load_config', dict()),
    )
    model.load_state_dict(
        checkpoint['model'],
        **kwargs.get('model_config', dict()),
    )
    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint['optimizer'],
            **kwargs.get('optimizer_config', dict()),
        )
    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint['scheduler'],
            **kwargs.get('scheduler_config', dict()),
        )
    return checkpoint.get('meta')
