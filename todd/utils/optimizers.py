__all__ = [
    'OPTIMIZERS',
    'build_param_group',
    'build_param_groups',
    'build_optimizer',
]

import inspect
import re
from functools import partial
from typing import Any, Dict, List, Optional, Sequence

import torch.nn as nn
import torch.optim as optim

from ..base import Config, Registry, default_build


def build_param_group(
    model: nn.Module,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    params = dict(params)
    regex = re.compile(params['params'])
    params['params'] = []
    for n, p in model.named_parameters():
        if regex.match(n):
            params['params'].append(p)
    return params


def build_param_groups(
    model: nn.Module,
    params: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if params is None:
        return [dict(params=model.parameters())]
    builder = partial(build_param_group, model)
    return list(map(builder, params))


def build_optimizer(
    registry: Registry[optim.Optimizer],
    config: Config,
) -> optim.Optimizer:
    model = config.pop('model')
    config.params = build_param_groups(model, config.get('params', None))
    return default_build(registry, config)


OPTIMIZERS = Registry(
    'optimizers',
    base=optim.Optimizer,
    build_func=build_optimizer,
)
for _, class_ in inspect.getmembers(optim, inspect.isclass):
    assert issubclass(class_, optim.Optimizer)
    OPTIMIZERS.register(class_)
