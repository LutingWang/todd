__all__ = [
    'load_open_mmlab_models',
    'transfer_weight',
    'transfer_weights',
    'save_checkpoint',
    'load_checkpoint',
]

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ._extensions import Module, get_logger, getattr_recur
from .configs import Config
from .registries import Registry


def load_open_mmlab_models(
    registry: Registry,
    config: str,
    config_options: Optional[Config] = None,
    ckpt: Optional[str] = None,
) -> Module:
    import mmcv
    import mmcv.runner

    config_dict = mmcv.Config.fromfile(config).model
    model = (
        registry.build(config_dict) if config_options is None else
        registry.build(config_options, default_args=config_dict)
    )
    if ckpt is not None:
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
    target._is_init = True


def transfer_weights(models, weight_prefixes: Dict[str, str]) -> None:
    for target_prefix, source_prefix in weight_prefixes.items():
        target = getattr_recur(models, target_prefix)
        source = getattr_recur(models, source_prefix)
        transfer_weight(target, source)


def save_checkpoint(
    model: nn.Module,
    f: str,
    *,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    meta: Optional[Config] = None,
    **kwargs,
) -> None:
    # TODO: state dict to cpu
    checkpoint: Dict[str, Any] = dict()
    checkpoint['model'] = model.state_dict()
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    if meta is not None:
        checkpoint['meta'] = meta
    torch.save(checkpoint, f, **kwargs)


def load_checkpoint(
    model: nn.Module,
    f: str,
    *,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    **kwargs,
) -> Optional[Config]:
    checkpoint: Dict[str, Any] = torch.load(f, **kwargs)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
    return checkpoint.get('meta')
