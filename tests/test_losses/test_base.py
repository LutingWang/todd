from typing import Callable, cast

import torch

from todd.losses.base import BaseLoss


class Loss(BaseLoss):

    def forward(self, loss: torch.Tensor, coef: float = 1.0) -> torch.Tensor:
        return loss * coef


class TestBase:

    def test_weight(self):
        loss = torch.tensor(1.0)

        loss_module = cast(Callable[..., torch.Tensor], Loss())
        assert loss_module(loss).item() == 1.0

        loss_module = cast(Callable[..., torch.Tensor], Loss(weight=0.5))
        assert loss_module(loss).item() == 0.5

    def test_bound(self):
        loss_module = cast(Callable[..., torch.Tensor], Loss(bound=1.0))
        loss = torch.tensor(1.0, requires_grad=True)
        bounded_loss = loss_module(loss)
        assert bounded_loss.item() == 1.0
        bounded_loss.backward()
        assert loss.grad.item() == 1.0

        loss_module = cast(Callable[..., torch.Tensor], Loss(bound=0.5))
        loss = torch.tensor(1.0, requires_grad=True)
        bounded_loss = loss_module(loss)
        assert bounded_loss.item() == 0.5
        bounded_loss.backward()
        assert loss.grad.item() == 0.5

        loss_module = cast(
            Callable[..., torch.Tensor],
            Loss(weight=2.0, bound=1.0),
        )
        loss = torch.tensor(1.0, requires_grad=True)
        bounded_loss = loss_module(loss)
        assert bounded_loss.item() == 1.0
        bounded_loss.backward()
        assert loss.grad.item() == 1.0
