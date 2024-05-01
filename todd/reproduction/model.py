__all__ = [
    'CheckMixin',
    'NoGradMixin',
    'EvalMixin',
    'FreezeMixin',
    'FrozenMixin',
]

from abc import ABC, abstractmethod
from typing import Any, Generator, MutableMapping
from typing_extensions import Self

import torch
from torch import nn

from ..base import BuildSpec, Config, FilterRegistry, Store
from ..base.registries import BuildSpecMixin
from ..models import InitWeightsMixin
from ..utils import classproperty
from .filters import NamedModulesFilter, NamedParametersFilter


class CheckMixin(nn.Module, ABC):
    """Mixin to perform a check before the first forward pass of a module.

    You should implement `check` in your subclass:

        >>> class Model(CheckMixin):
        ...     def check(self, module: Self, *args, **kwargs) -> None:
        ...         print(
        ...             f"Checking {repr(module.__class__.__name__)} with "
        ...             f"{args=} and {kwargs=}"
        ...         )
        ...     def forward(self, *args, **kwargs) -> None:
        ...         print(f"Forwarding with {args=} and {kwargs=}")

    The `check` method executes prior to ``__call__``:

        >>> check = Model()
        >>> check(1, 2, a=3, b=4)
        Checking 'Model' with args=(1, 2) and kwargs={'a': 3, 'b': 4}
        Forwarding with args=(1, 2) and kwargs={'a': 3, 'b': 4}

    If `Store.DRY_RUN` is False, the `check` method executes only once:

        >>> Store.DRY_RUN
        False
        >>> check(1, a=2)
        Forwarding with args=(1,) and kwargs={'a': 2}

    If `Store.DRY_RUN` is True, the `check` method executes every time
    ``__call__`` is invoked:

        >>> Store.DRY_RUN = True
        >>> check = Model()
        >>> check(1, 2, a=3, b=4)
        Checking 'Model' with args=(1, 2) and kwargs={'a': 3, 'b': 4}
        Forwarding with args=(1, 2) and kwargs={'a': 3, 'b': 4}
        >>> check(1, a=2)
        Checking 'Model' with args=(1,) and kwargs={'a': 2}
        Forwarding with args=(1,) and kwargs={'a': 2}
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        def forward_pre_hook(
            module: Self,
            args: tuple,
            kwargs: dict[str, Any],
        ) -> None:
            self.check(module, *args, **kwargs)
            if not Store.DRY_RUN:
                handle.remove()

        handle = self.register_forward_pre_hook(
            forward_pre_hook,
            with_kwargs=True,
        )

    @abstractmethod
    def check(self, module: Self, *args, **kwargs) -> None:
        pass


class NoGradMixin(CheckMixin, InitWeightsMixin, BuildSpecMixin):

    def __init__(
        self,
        *args,
        no_grad_filter: NamedParametersFilter | None = None,
        filter_state_dict: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._no_grad_filter = no_grad_filter
        self._filter_state_dict = filter_state_dict

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec = BuildSpec(no_grad_filter=FilterRegistry.build)
        return super().build_spec | build_spec

    def check(self, module: Self, *args, **kwargs) -> None:
        super().check(module, *args, **kwargs)
        for _, parameter in module._no_grad_named_parameters():
            assert not parameter.requires_grad

    def _no_grad_named_parameters(
        self,
    ) -> Generator[tuple[str, nn.Parameter], None, None]:
        if self._no_grad_filter is not None:
            yield from self._no_grad_filter(self)

    def init_weights(self, config: Config) -> bool:
        self.requires_grad_()
        return super().init_weights(config)

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        super().requires_grad_(requires_grad)
        for _, parameter in self._no_grad_named_parameters():
            parameter.requires_grad_(False)
        return self

    def state_dict(self, *args, prefix: str = '', **kwargs):
        """Remove frozen parameters from `torch.nn.Module.load_state_dict`.

        Define a ``no_grad_filter``:

            >>> class Model(NoGradMixin):
            ...     def __init__(self, *args, **kwargs) -> None:
            ...         super().__init__(
            ...             *args,
            ...             no_grad_filter=NamedParametersFilter(
            ...                 name='_conv.weight',
            ...             ),
            ...             filter_state_dict=True,
            ...             **kwargs,
            ...         )
            ...         self._conv = nn.Conv1d(1, 2, 1)

        The parameters without gradient will be excluded from the
        ``state_dict``:

            >>> model = Model()
            >>> model.state_dict()
            OrderedDict([('_conv.bias', tensor([..., ...]))])

        The model can even be used as building blocks in other models:

            >>> sequential = nn.Sequential(model)
            >>> sequential.state_dict()
            OrderedDict([('0._conv.bias', tensor([..., ...]))])
        """
        state_dict: MutableMapping[str, torch.Tensor] = \
            super().state_dict(*args, prefix=prefix, **kwargs)
        for name, _ in self._no_grad_named_parameters():
            state_dict.pop(prefix + name, None)
        return state_dict


class EvalMixin(CheckMixin, InitWeightsMixin, BuildSpecMixin):

    def __init__(
        self,
        *args,
        eval_filter: NamedModulesFilter | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._eval_filter = eval_filter

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec = BuildSpec(eval_filter=FilterRegistry.build)
        return super().build_spec | build_spec

    def check(self, module: Self, *args, **kwargs) -> None:
        super().check(module, *args, **kwargs)
        for _, module_ in module._eval_modules():
            assert not module_.training

    def _eval_modules(self) -> Generator[tuple[str, nn.Module], None, None]:
        if self._eval_filter is not None:
            yield from self._eval_filter(self)

    def init_weights(self, config: Config) -> bool:
        self.train()
        return super().init_weights(config)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        for _, module in self._eval_modules():
            if module is self:
                super().eval()
            else:
                module.eval()
        return self


class FreezeMixin(NoGradMixin, EvalMixin):  # type: ignore[misc]
    pass


class FrozenMixin(FreezeMixin):
    pass
