import pytest
import torch
from torch import nn

import todd
from todd.reproduction.model import FrozenMixin


class InitWeightsMixin:

    def init_weights(self, config: todd.Config) -> bool:
        self._initialized = True
        return config.recursive


class Model(FrozenMixin, InitWeightsMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(128, 256, 3, bias=False)
        self.bn = nn.BatchNorm2d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class TestFrozenMixin:

    def test_init_weights(self) -> None:
        model = Model(
            no_grad_filter=dict(
                type='NamedParametersFilter',
                modules=dict(
                    type='NamedModulesFilter',
                    name='conv',
                ),
            ),
            eval_filter=dict(
                type='NamedModulesFilter',
                name='bn',
            ),
        )

        assert not hasattr(model, '_initialized')
        assert model.conv.weight.requires_grad
        assert model.bn.training
        assert model.init_weights(todd.Config(recursive=True))
        assert model._initialized
        assert not model.conv.weight.requires_grad
        assert not model.bn.training

        assert not model.init_weights(todd.Config(recursive=False))

    def test_check_no_grad(self) -> None:
        model = Model(
            no_grad_filter=dict(
                type='NamedParametersFilter',
                modules=dict(
                    type='NamedModulesFilter',
                    name='conv',
                ),
            ),
        )
        sequential = nn.Sequential(model)

        model.requires_grad_(True)
        assert not model.conv.weight.requires_grad

        sequential.requires_grad_(True)
        assert model.conv.weight.requires_grad

        x = torch.rand(1, 128, 32, 32)
        with pytest.raises(AssertionError):
            model(x)

    def test_check_eval(self) -> None:
        model = Model(
            eval_filter=dict(
                type='NamedModulesFilter',
                name='bn',
            ),
        )
        sequential = nn.Sequential(model)

        model.train()
        assert not model.bn.training

        sequential.train()
        assert not model.bn.training

    def test_requires_grad(self) -> None:
        model = Model(
            no_grad_filter=dict(
                type='NamedParametersFilter',
                modules=dict(
                    type='NamedModulesFilter',
                    name='conv',
                ),
            ),
        )

        model.requires_grad_(True)
        assert not model.conv.weight.requires_grad
        assert model.bn.weight.requires_grad
        assert model.bn.bias.requires_grad

        model.requires_grad_(False)
        assert not model.conv.weight.requires_grad
        assert not model.bn.weight.requires_grad
        assert not model.bn.bias.requires_grad

    def test_train(self) -> None:
        model = Model(
            eval_filter=dict(
                type='NamedModulesFilter',
                name='bn',
            ),
        )

        model.train()
        assert model.conv.training
        assert not model.bn.training

        model.eval()
        assert not model.conv.training
        assert not model.bn.training
