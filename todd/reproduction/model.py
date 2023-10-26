__all__ = [
    'FrozenMixin',
]

from abc import ABC
from typing import Any, Generator, MutableMapping
from typing_extensions import Self

from torch import nn

from ..base import (
    Config,
    FilterRegistry,
    NamedModulesFilter,
    NamedParametersFilter,
    Store,
)


class FrozenMixin(nn.Module, ABC):

    def __init__(
        self,
        *args,
        no_grad_filter: Config | None = None,
        eval_filter: Config | None = None,
        filter_state_dict: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._no_grad_filter: NamedParametersFilter | None = (
            None if no_grad_filter is None else
            FilterRegistry.build(no_grad_filter)
        )
        self._eval_filter: NamedModulesFilter | None = (
            None if eval_filter is None else FilterRegistry.build(eval_filter)
        )
        self._filter_state_dict = filter_state_dict

        self.check_no_grad()
        self.check_eval()
        if filter_state_dict:
            self._register_state_dict_hook(self.state_dict_hook)

    def check_no_grad(self) -> None:

        def forward_pre_hook(module: 'FrozenMixin', *args, **kwargs) -> None:
            for _, parameter in module._no_grad_named_parameters():
                assert not parameter.requires_grad
            if not Store.DRY_RUN:
                handle.remove()

        handle = self.register_forward_pre_hook(forward_pre_hook)

    def check_eval(self) -> None:

        def forward_pre_hook(module: 'FrozenMixin', *args, **kwargs) -> None:
            for _, module_ in module._eval_modules():
                assert not module_.training
            if not Store.DRY_RUN:
                handle.remove()

        handle = self.register_forward_pre_hook(forward_pre_hook)

    def _no_grad_named_parameters(
        self,
    ) -> Generator[tuple[str, nn.Parameter], None, None]:
        if self._no_grad_filter is not None:
            yield from self._no_grad_filter(self)

    def _eval_modules(self) -> Generator[tuple[str, nn.Module], None, None]:
        if self._eval_filter is not None:
            yield from self._eval_filter(self)

    def init_weights(self, config: Config) -> bool:
        recursive = True
        if hasattr(super(), 'init_weights'):
            recursive = super().init_weights(config)  # type: ignore[misc]
        self.requires_grad_()
        self.train()
        return recursive

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        super().requires_grad_(requires_grad)
        for _, parameter in self._no_grad_named_parameters():
            parameter.requires_grad_(False)
        return self

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        for _, module in self._eval_modules():
            module.eval()
        return self

    @staticmethod
    def state_dict_hook(
        model: 'FrozenMixin',
        state_dict: MutableMapping[str, Any],
        prefix: str,
        *args,
        **kwargs,
    ) -> None:
        """Remove frozen parameters from `torch.nn.Module.load_state_dict`.

        Args:
            model: the model to load state dict.
            state_dict: the state dict to load.
            prefix: the prefix of the model.
            *args: other args.
            **kwargs: other kwargs.

        Example:
            >>> class Model(FrozenMixin):
            ...     def __init__(self, *args, **kwargs) -> None:
            ...         super().__init__(
            ...             *args,
            ...             no_grad_filter=dict(
            ...                 type='NamedParametersFilter',
            ...                 modules=dict(
            ...                     type='NamedModulesFilter',
            ...                     name='conv',
            ...                 ),
            ...             ),
            ...             filter_state_dict=True,
            ...             **kwargs,
            ...         )
            ...         self.conv = nn.Conv1d(1, 2, 1)
            ...         self.bn = nn.BatchNorm1d(2)
            >>> Model().state_dict()
            OrderedDict([('bn.weight', tensor([1., 1.])), ('bn.bias', tensor([\
0., 0.])), ('bn.running_mean', tensor([0., 0.])), ('bn.running_var', tensor([\
1., 1.])), ('bn.num_batches_tracked', tensor(0))])
            >>> nn.Sequential(Model()).state_dict()
            OrderedDict([('0.bn.weight', tensor([1., 1.])), ('0.bn.bias', \
tensor([0., 0.])), ('0.bn.running_mean', tensor([0., 0.])), (\
'0.bn.running_var', tensor([1., 1.])), ('0.bn.num_batches_tracked', \
tensor(0))])
        """
        for name, _ in model._no_grad_named_parameters():
            state_dict.pop(prefix + name, None)
