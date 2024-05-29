# pylint: disable=abstract-class-instantiated

import torch

from todd.models.losses.base import BaseLoss


class Loss(BaseLoss):

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return torch.tensor(1.0)


class TestBase:

    def test_weight(self) -> None:
        assert Loss()().item() == 1.0
        assert Loss(weight=0.5)().item() == 0.5

    def test_bound(self):
        assert Loss(bound=1.0)().item() == 1.0
        assert Loss(bound=0.5)().item() == 0.5
        assert Loss(weight=2.0, bound=1.0)().item() == 1.0
