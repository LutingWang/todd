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

from ..bases.configs import Config
from ..bases.registries import BuildPreHookMixin, Item, RegistryMeta
from ..registries import InitWeightsMixin
from ..utils import Store
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
        ...             f"Checking {module.__class__.__name__!r} with "
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


# `CheckMixin` uses `nn.Module` as base class, so it should be the last mixin,
# in order to avoid MRO resolution failures.


class NoGradMixin(InitWeightsMixin, BuildPreHookMixin, CheckMixin):
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

    Given a model that inherits from this mixin class:

        >>> class NoGrad(NoGradMixin):
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...         self.conv = nn.Conv2d(1, 2, 3)
        ...     def forward(self) -> None:
        ...         pass

    Users can specify parameters to be excluded from gradient computation via
    a filter:

        >>> npf = NamedParametersFilter(name='conv.weight')
        >>> model = NoGrad(no_grad=npf)

    The exclusion of parameters from gradient computation does not happen
    immediately after the model is constructed:

        >>> {n: p.requires_grad for n, p in model.named_parameters()}
        {'conv.weight': True, 'conv.bias': True}

    Instead, the exclusion is triggered by calling `init_weights`:

        >>> _ = model.init_weights(Config())
        >>> {n: p.requires_grad for n, p in model.named_parameters()}
        {'conv.weight': False, 'conv.bias': True}

    or by calling `requires_grad_`:

        >>> _ = model.requires_grad_()
        >>> {n: p.requires_grad for n, p in model.named_parameters()}
        {'conv.weight': False, 'conv.bias': True}
        >>> _ = model.requires_grad_(False)
        >>> {n: p.requires_grad for n, p in model.named_parameters()}
        {'conv.weight': False, 'conv.bias': False}

    Note that parameters that should be excluded from gradient computation can
    sometimes be included.
    A typical example is when the model is used as a component in another:

        >>> sequential = nn.Sequential(model)
        >>> _ = sequential.requires_grad_()
        >>> {n: p.requires_grad for n, p in sequential.named_parameters()}
        {'0.conv.weight': True, '0.conv.bias': True}

    To prevent this, the `check` method can be used to verify if parameters
    are correctly excluded from gradient computation:

        >>> model.check(model, tuple(), dict())
        Traceback (most recent call last):
            ...
        RuntimeError: conv.weight requires grad
        >>> _ = model.requires_grad_()
        >>> model.check(model, tuple(), dict())

    Refer to `CheckMixin` for more information on the `check` method.

    By default, the state dictionary includes all parameters:

        >>> list(model.state_dict())
        ['conv.weight', 'conv.bias']

    However, in most cases, users may want to exclude frozen parameters from
    the state dictionary.
    This can be achieved by setting ``filter_state_dict`` to True:

        >>> model = NoGrad(no_grad=npf, filter_state_dict=True)
        >>> list(model.state_dict())
        ['conv.bias']

    State dictionary filtering works even if the model is used as a component
    in another model:

        >>> sequential = nn.Sequential(model)
        >>> list(sequential.state_dict())
        ['0.conv.bias']
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

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        if 'no_grad' in config:
            config.no_grad = FilterRegistry.build_or_return(config.no_grad)
        return config

    def check(self, *args, **kwargs) -> None:
        super().check(*args, **kwargs)  # type: ignore[safe-super]
        for name, parameter in self._no_grad_named_parameters():
            if parameter.requires_grad:
                raise RuntimeError(f"{name} requires grad")

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
        if self._filter_state_dict:
            for name, _ in self._no_grad_named_parameters():
                state_dict.pop(prefix + name, None)
        return state_dict


class EvalMixin(InitWeightsMixin, BuildPreHookMixin, CheckMixin):
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

        >>> model.check(model, tuple(), dict())
        >>> model.bn.training = True
        >>> model.check(model, tuple(), dict())
        Traceback (most recent call last):
            ...
        RuntimeError: bn is in training mode

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

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        if 'eval_' in config:
            config.eval_ = FilterRegistry.build_or_return(config.eval_)
        return config

    def check(self, *args, **kwargs) -> None:
        super().check(*args, **kwargs)  # type: ignore[safe-super]
        for name, module in self._eval_modules():
            if module.training:
                raise RuntimeError(f"{name} is in training mode")

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
    """A mixin class that provides freezing functionality to a model.

    Examples:
        >>> class Freeze(FreezeMixin):
        ...     def __init__(self, *args, **kwargs) -> None:
        ...         super().__init__(*args, **kwargs)
        ...         self.c = nn.Conv2d(1, 2, 3)
        ...         self.b = nn.BatchNorm2d(2)
        >>> nmf = NamedModulesFilter(name='b')
        >>> freeze = Freeze(freeze=nmf)
        >>> freeze.init_weights(Config())
        True
        >>> {n: p.requires_grad for n, p in freeze.named_parameters()}
        {'c.weight': True, 'c.bias': True, 'b.weight': False, 'b.bias': False}
        >>> {n: m.training for n, m in freeze.named_modules()}
        {'': True, 'c': True, 'b': False}
    """

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

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        if 'freeze' in config:
            config.freeze = FilterRegistry.build_or_return(config.freeze)
        return config


class FrozenMixin(FreezeMixin):
    """A mixin class that provides freezing functionality to a class.

    This mixin class is used to create frozen modules, where the parameters
    are excluded from gradient computation and the modules are marked as
    evaluation.

    Examples:
        >>> class Frozen(FrozenMixin):
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.conv = nn.Conv2d(1, 2, 3)
        >>> frozen = Frozen()
        >>> frozen.init_weights(Config())
        True
        >>> {n: p.requires_grad for n, p in frozen.named_parameters()}
        {'conv.weight': False, 'conv.bias': False}
        >>> {n: m.training for n, m in frozen.named_modules()}
        {'': False, 'conv': False}
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, freeze=NamedModulesFilter(name=''), **kwargs)
