from typing import cast

import pytest
from torch import nn

from todd import Config
from todd.reproduction.model import FrozenMixin


class Model(FrozenMixin):

    def __init__(self, *args, **kwargs) -> None:
        kwargs.update(
            requires_grad_configs=[
                Config(names=['.conv'], mode=False),
            ],
            train_configs=[
                Config(names=['.bn'], mode=False),
            ],
        )
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(128, 256, 3)
        self.bn = nn.BatchNorm2d(256)


class TestFrozenMixin:

    @pytest.fixture
    def model(self) -> Model:
        return Model()

    def test_requires_grad(self, model: Model) -> None:
        bias = cast(nn.Parameter, model.conv.bias)

        model.requires_grad_(True)
        assert not model.conv.weight.requires_grad
        assert not bias.requires_grad
        assert model.bn.weight.requires_grad
        assert model.bn.bias.requires_grad

        model.requires_grad_(False)
        assert not model.conv.weight.requires_grad
        assert not bias.requires_grad
        assert not model.bn.weight.requires_grad
        assert not model.bn.bias.requires_grad

    def test_train(self, model: Model) -> None:
        model.train()
        assert model.conv.training
        assert not model.bn.training

        model.eval()
        assert not model.conv.training
        assert not model.bn.training
