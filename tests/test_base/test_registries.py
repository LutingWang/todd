from typing import cast

import pytest
import torch
from custom_types import CustomModule

from todd import Config
from todd.base.registries import (
    OptimizerRegistry,
    Registry,
    RegistryMeta,
    build_param_group,
    build_param_groups,
)


class Registry1(metaclass=RegistryMeta):
    pass


class Registry2(Registry1):
    pass


class Registry3(Registry2):
    pass


class Registry3_1(Registry2):
    pass


class TestRegistryMeta:

    def test_call(self) -> None:
        with pytest.raises(TypeError):
            Registry()

    def test_missing(self) -> None:
        key = 'custom_key'
        with pytest.raises(KeyError, match=key):
            Registry[key]

    def test_parse(self) -> None:
        registry, key = Registry1._parse('Registry2.Registry3.custom_key')
        assert registry is Registry3
        assert key == 'custom_key'

    def test_child(self) -> None:
        with pytest.raises(ValueError):
            Registry1.child('Registry1')
        with pytest.raises(ValueError):
            Registry1.child('Registry2.Registry2')

        class Registry3_1(Registry2):
            pass

        with pytest.raises(ValueError):
            Registry1.child('Registry2.Registry3_1')

    def test_build(self) -> None:
        with pytest.raises(KeyError):
            Registry.build(Config(type='custom_key'))


def test_build_param_group(model: CustomModule) -> None:
    params = build_param_group(model, Config(params='conv.[^w]'))
    assert len(params) == 1
    assert len(params['params']) == 1
    param = cast(torch.Tensor, model.conv.bias)
    assert param.eq(params['params'][0]).all()


def test_build_param_groups(model: CustomModule) -> None:
    params = build_param_groups(model)
    assert len(params) == 1
    assert len(params[0]) == 1
    assert len(list(params[0]['params'])) == len(list(model.parameters()))

    params = build_param_groups(model, [Config(params='conv.[^w]')])
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
    assert isinstance(optimizer, torch.optim.Adam)
