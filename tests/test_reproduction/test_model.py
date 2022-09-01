import torch.nn as nn
from custom_types import CustomModule

from todd.reproduction.model import (
    FrozenMixin,
    eval_,
    freeze,
    get_modules,
    no_grad,
)


def test_get_modules(model: CustomModule) -> None:
    assert get_modules(model) == [model]
    assert get_modules(model, names=['conv']) == [model.conv]
    assert get_modules(model, types=[nn.Conv2d]) == [model.conv]
    assert get_modules(
        model,
        names=['conv', 'module'],
        types=[CustomModule],
    ) == [model.module]


def test_no_grad(model: CustomModule) -> None:
    no_grad(model, names=['module'])
    for param in model.conv.parameters():
        assert param.requires_grad
    for param in model.module.parameters():
        assert not param.requires_grad


def test_eval(model: CustomModule) -> None:
    eval_(model, names=['module'])
    assert model.conv.training
    for module in model.module.modules():
        assert not module.training


def test_freeze(model: CustomModule) -> None:
    freeze(model, names=['module'])
    for param in model.conv.parameters():
        assert param.requires_grad
    for param in model.module.parameters():
        assert not param.requires_grad
    assert model.conv.training
    for module in model.module.modules():
        assert not module.training


class FrozenModule(FrozenMixin, nn.Module):

    def __init__(self, **kwargs) -> None:
        nn.Module.__init__(self)
        self._no_grad_module = nn.Conv2d(128, 256, 3)
        self._eval_module = nn.Linear(1024, 10)
        FrozenMixin.__init__(self, **kwargs)


class TestFrozenMixin:

    def test_frozen(self) -> None:
        model = FrozenModule(
            no_grad_config=dict(names=['_no_grad_module']),
            eval_config=dict(types=[nn.Linear]),
        )

        for param in model._no_grad_module.parameters():
            assert not param.requires_grad
        for param in model._eval_module.parameters():
            assert param.requires_grad
        assert model._no_grad_module.training
        assert not model._eval_module.training

        model.requires_grad_(False)
        for param in model.parameters():
            assert not param.requires_grad
        model.requires_grad_(True)
        for param in model._no_grad_module.parameters():
            assert not param.requires_grad
        for param in model._eval_module.parameters():
            assert param.requires_grad

        model.eval()
        for module in model.modules():
            assert not module.training
        model.train()
        assert model._no_grad_module.training
        assert not model._eval_module.training

    def test_normal(self) -> None:
        model = FrozenModule()

        for param in model.parameters():
            assert param.requires_grad
        for module in model.modules():
            assert module.training

        model.requires_grad_(False)
        for param in model.parameters():
            assert not param.requires_grad
        model.requires_grad_(True)
        for param in model.parameters():
            assert param.requires_grad

        model.eval()
        for module in model.modules():
            assert not module.training
        model.train()
        for module in model.modules():
            assert module.training
