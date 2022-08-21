from custom_types import CustomModule

from todd.reproduction.model import EvalMode, NoGradMode


class TestNoGradMode:

    def test_all(self, model: CustomModule) -> None:
        NoGradMode.ALL.no_grad(model)
        for param in model.parameters():
            assert not param.requires_grad

    def test_none(self, model: CustomModule) -> None:
        NoGradMode.NONE.no_grad(model)
        for param in model.parameters():
            assert param.requires_grad

    def test_partial(self, model: CustomModule) -> None:
        NoGradMode.PARTIAL.no_grad(model, module_names=['module'])
        for param in model.conv.parameters():
            assert param.requires_grad
        for param in model.module.parameters():
            assert not param.requires_grad


class TestEvalMode:

    def test_all(self, model: CustomModule) -> None:
        EvalMode.ALL.eval(model)
        for module in model.modules():
            assert not module.training

    def test_none(self, model: CustomModule) -> None:
        EvalMode.NONE.eval(model)
        for module in model.modules():
            assert module.training
