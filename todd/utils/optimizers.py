__all__ = [
    'LrSchedulerRegistry',
    'OptimizerRegistry',
    'build_param_group',
    'build_param_groups',
]

import inspect
import re
from functools import partial
from typing import Any, Sequence

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..base import Config, Registry, RegistryMeta


def build_param_group(
    model: nn.Module,
    params: dict[str, Any],
) -> dict[str, Any]:
    params = dict(params)
    regex = re.compile(params['params'])
    params['params'] = []
    for n, p in model.named_parameters():
        if regex.match(n):
            params['params'].append(p)
    return params


def build_param_groups(
    model: nn.Module,
    params: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if params is None:
        return [dict(params=model.parameters())]
    builder = partial(build_param_group, model)
    return list(map(builder, params))


class OptimizerRegistry(Registry):

    @classmethod
    def _build(cls, config: Config) -> optim.Optimizer:
        model = config.pop('model')
        config.params = build_param_groups(model, config.get('params', None))
        return RegistryMeta._build(cls, config)


for _, class_ in inspect.getmembers(optim, inspect.isclass):
    assert issubclass(class_, optim.Optimizer)
    OptimizerRegistry.register()(class_)


class LrSchedulerRegistry(Registry):
    pass


for _, class_ in inspect.getmembers(optim.lr_scheduler, inspect.isclass):
    if issubclass(class_, lr_scheduler._LRScheduler):
        LrSchedulerRegistry.register()(class_)
