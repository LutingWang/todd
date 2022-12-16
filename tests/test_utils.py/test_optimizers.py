from typing import cast

import torch
import torch.optim as optim
from custom_types import CustomModule

from todd.base import Config
from todd.utils.optimizers import (
    OptimizerRegistry,
    build_param_group,
    build_param_groups,
)


def test_build_param_group(model: CustomModule) -> None:
    params = build_param_group(model, dict(params='conv.[^w]'))
    assert len(params) == 1
    assert len(params['params']) == 1
    param = cast(torch.Tensor, model.conv.bias)
    assert param.eq(params['params'][0]).all()


def test_build_param_groups(model: CustomModule) -> None:
    params = build_param_groups(model)
    assert len(params) == 1
    assert len(params[0]) == 1
    assert len(list(params[0]['params'])) == len(list(model.parameters()))

    params = build_param_groups(model, [dict(params='conv.[^w]')])
    assert len(params) == 1
    assert len(params[0]) == 1
    assert len(params[0]['params']) == 1
    param = cast(torch.Tensor, model.conv.bias)
    assert param.eq(params[0]['params'][0]).all()


def test_build_optimizer(model: CustomModule) -> None:
    optimizer = OptimizerRegistry._build(
        Config(
            type='Adam',
            model=model,
            params=[dict(params='conv.[^w]')],
        ),
    )
    assert isinstance(optimizer, optim.Adam)
