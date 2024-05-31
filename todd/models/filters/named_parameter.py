__all__ = [
    'NamedParametersFilter',
]

from typing import Generator, Iterable

from torch import nn

from ...patches.py import classproperty
from ...registries import BuildSpec, BuildSpecMixin
from ..registries import FilterRegistry
from .named_member import NamedMembersFilter
from .named_module import NamedModulesFilter


@FilterRegistry.register_()
class NamedParametersFilter(NamedMembersFilter[nn.Parameter], BuildSpecMixin):

    def __init__(
        self,
        *args,
        modules: NamedModulesFilter | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert self._names is None or modules is None
        self._named_modules_filter = modules

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec = BuildSpec(modules=FilterRegistry.build)
        return super().build_spec | build_spec

    @property
    def named_modules_filter(self) -> NamedModulesFilter:
        assert self._named_modules_filter is not None
        return self._named_modules_filter

    def _named_members(
        self,
        module: nn.Module,
    ) -> Generator[tuple[str, nn.Parameter], None, None]:
        named_modules: Iterable[tuple[str, nn.Module]]
        if self._named_modules_filter is None:
            named_modules = [('', module)]
        else:
            named_modules = self.named_modules_filter(module)
        for module_name, module_ in named_modules:
            if module_name != '':
                module_name += '.'
            for parameter_name, parameter in module_.named_parameters():
                yield module_name + parameter_name, parameter
