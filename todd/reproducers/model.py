__all__ = [
    'CheckMixin',
    'NoGradMixin',
    'EvalMixin',
    'FreezeMixin',
    'FrozenMixin',
]

from abc import ABC, abstractmethod
from typing import Any, Iterable, MutableMapping
from typing_extensions import Self

import torch
from torch import nn

from ..configs import Config
from ..models import InitWeightsMixin
from ..patches import classproperty
from ..registries import BuildSpec, BuildSpecMixin
from ..stores import Store
from .filters import NamedModulesFilter, NamedParametersFilter
from .registries import FilterRegistry


class CheckMixin(nn.Module, ABC):
    """Mixin to perform a check before the first forward pass of a module.

    You should implement `check` in your subclass:

        >>> class Model(CheckMixin):
        ...     def check(
        ...         self,
        ...         module: nn.Module,
        ...         args: tuple[Any],
        ...         kwargs: dict[str, Any],
        ...     ) -> None:
        ...         print(
        ...             f"Checking {repr(module.__class__.__name__)} with "
        ...             f"{args=} and {kwargs=}"
        ...         )
        ...     def forward(self, *args, **kwargs) -> None:
        ...         print(f"Forwarding with {args=} and {kwargs=}")

    The `check` method executes prior to ``__call__``:

        >>> model = Model()
        >>> model(1, 2, a=3, b=4)
        Checking 'Model' with args=(1, 2) and kwargs={'a': 3, 'b': 4}
        Forwarding with args=(1, 2) and kwargs={'a': 3, 'b': 4}

    If `Store.DRY_RUN` is False, the `check` method executes only once:

        >>> Store.DRY_RUN
        False
        >>> model(1, a=2)
        Forwarding with args=(1,) and kwargs={'a': 2}

    If `Store.DRY_RUN` is True, the `check` method executes every time
    ``__call__`` is invoked:

        >>> Store.DRY_RUN = True
        >>> model = Model()
        >>> model(1, 2, a=3, b=4)
        Checking 'Model' with args=(1, 2) and kwargs={'a': 3, 'b': 4}
        Forwarding with args=(1, 2) and kwargs={'a': 3, 'b': 4}
        >>> model(1, a=2)
        Checking 'Model' with args=(1,) and kwargs={'a': 2}
        Forwarding with args=(1,) and kwargs={'a': 2}
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        def forward_pre_hook(*args, **kwargs) -> None:
            self.check(*args, **kwargs)
            if not Store.DRY_RUN:
                handle.remove()

        handle = self.register_forward_pre_hook(
            forward_pre_hook,
            with_kwargs=True,
        )

    @abstractmethod
    def check(
        self,
        module: nn.Module,
        args: tuple[Any],
        kwargs: dict[str, Any],
    ) -> None:
        pass


class NoGradMixin(CheckMixin, InitWeightsMixin, BuildSpecMixin):
    """A mixin class that excludes specific parameters from gradient \
    computation.

    This mixin class is designed to be a base class for models that necessitate
    certain parameters to be excluded from gradient computation.
    It offers methods to scrutinize the model's weights and modify the state
    dictionary for excluding frozen parameters.

    Args:
        no_grad: A function specifying the parameters to be
            excluded from gradient computation.
        filter_state_dict: A flag controlling whether to filter the
            state dictionary to exclude frozen parameters.

    To use the mixin class, first define a model:

        >>> class NoGrad(NoGradMixin):
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...         self.conv = nn.Conv2d(1, 2, 3)
        ...     def forward(self) -> None:
        ...         pass

    Set up a parameter filter:

        >>> npf = NamedParametersFilter(name='conv.weight')

    Construct the model:

        >>> model = NoGrad(no_grad=npf, filter_state_dict=True)
        >>> model.conv.weight.requires_grad
        True
        >>> model.conv.bias.requires_grad
        True

    Use `init_weights` to change the gradient requirement:

        >>> model.init_weights(Config())
        True
        >>> model.conv.weight.requires_grad
        False
        >>> model.conv.bias.requires_grad
        True

    Use `check` method to verify if the properties meet the requirement:

        >>> model.check(model, tuple(), {})
        >>> model.conv.weight.requires_grad = True
        >>> model.check(model, tuple(), {})
        Traceback (most recent call last):
            ...
        AssertionError

    Use `requires_grad_` to enforce the properties to meet the requirement:

        >>> _ = model.requires_grad_()
        >>> model.conv.weight.requires_grad
        False
        >>> model.conv.bias.requires_grad
        True
        >>> _ = model.requires_grad_(False)
        >>> model.conv.weight.requires_grad
        False
        >>> model.conv.bias.requires_grad
        False

    Parameters that do not require gradient will be excluded from the state
    dictionary:

        >>> model.state_dict()
        OrderedDict([('conv.bias', tensor([..., ...]))])

    The model can be utilized as a component in other models:

        >>> sequential = nn.Sequential(model)
        >>> sequential.state_dict()
        OrderedDict([('0.conv.bias', tensor([..., ...]))])

    Calling `requires_grad_` on ``sequential`` could violate the
    requirements:

        >>> _ = sequential.requires_grad_()
        >>> sequential[0].conv.weight.requires_grad
        True
    """

    def __init__(
        self,
        *args,
        no_grad: NamedParametersFilter | None = None,
        filter_state_dict: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._no_grad = no_grad
        self._filter_state_dict = filter_state_dict

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec = BuildSpec(no_grad=FilterRegistry.build)
        return super().build_spec | build_spec

    def check(self, *args, **kwargs) -> None:
        super().check(*args, **kwargs)
        for _, parameter in self._no_grad_named_parameters():
            assert not parameter.requires_grad

    def _no_grad_named_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        if self._no_grad is None:
            return []
        return self._no_grad(self)

    def init_weights(self, config: Config) -> bool:
        self.requires_grad_()
        return super().init_weights(config)

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        super().requires_grad_(requires_grad)
        for _, parameter in self._no_grad_named_parameters():
            parameter.requires_grad_(False)
        return self

    def state_dict(self, *args, prefix: str = '', **kwargs) -> Any:
        state_dict: MutableMapping[str, torch.Tensor] = \
            super().state_dict(*args, prefix=prefix, **kwargs)
        for name, _ in self._no_grad_named_parameters():
            state_dict.pop(prefix + name, None)
        return state_dict


class EvalMixin(CheckMixin, InitWeightsMixin, BuildSpecMixin):
    """A mixin class that provides evaluation functionality for a model.

    This mixin class is intended to be used as a base class for models that
    require evaluation functionality.
    It provides methods for checking the model's evaluation mode and toggling
    between training and evaluation modes.

    Args:
        eval_: A function specifying the modules to be marked evaluation.

    To use the mixin class, first define a model:

        >>> class Eval(EvalMixin):
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...         self.conv = nn.Conv2d(1, 2, 3)
        ...         self.bn = nn.BatchNorm2d(2)
        ...     def forward(self) -> None:
        ...         pass

    Set up a module filter:

        >>> nmf = NamedModulesFilter(name='bn')

    Construct the model:

        >>> model = Eval(eval_=nmf)
        >>> model.conv.training
        True
        >>> model.bn.training
        True

    Use `init_weights` to change the training property:

        >>> model.init_weights(Config())
        True
        >>> model.conv.training
        True
        >>> model.bn.training
        False

    Use `check` method to verify if the properties meet the requirement:

        >>> model.check(model, tuple(), {})
        >>> model.bn.training = True
        >>> model.check(model, tuple(), {})
        Traceback (most recent call last):
            ...
        AssertionError

    Use `train` to enforce the properties to meet the requirement:

        >>> _ = model.train()
        >>> model.conv.training
        True
        >>> model.bn.training
        False
        >>> _ = model.eval()
        >>> model.conv.training
        False
        >>> model.bn.training
        False

    The model can be utilized as a component in other models:

        >>> sequential = nn.Sequential(model)
        >>> _ = sequential.train()
        >>> sequential[0].conv.training
        True
        >>> sequential[0].bn.training
        False
    """

    def __init__(
        self,
        *args,
        eval_: NamedModulesFilter | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._eval = eval_

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec = BuildSpec(eval_=FilterRegistry.build)
        return super().build_spec | build_spec

    def check(self, *args, **kwargs) -> None:
        super().check(*args, **kwargs)
        for _, module in self._eval_modules():
            assert not module.training

    def _eval_modules(self) -> Iterable[tuple[str, nn.Module]]:
        if self._eval is None:
            return []
        return self._eval(self)

    def init_weights(self, config: Config) -> bool:
        self.train()
        return super().init_weights(config)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        for _, module in self._eval_modules():
            if module is self:
                super().train(False)
            else:
                module.eval()
        return self


class FreezeMixin(NoGradMixin, EvalMixin):

    def __init__(
        self,
        *args,
        freeze: NamedModulesFilter | None = None,
        **kwargs,
    ) -> None:
        if freeze is None:
            super().__init__(*args, **kwargs)
            return
        super().__init__(
            *args,
            eval_=freeze,
            no_grad=NamedParametersFilter(modules=freeze),
            **kwargs,
        )

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec = BuildSpec(freeze=FilterRegistry.build)
        return super().build_spec | build_spec


class FrozenMixin(FreezeMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, freeze=NamedModulesFilter(name=''), **kwargs)
