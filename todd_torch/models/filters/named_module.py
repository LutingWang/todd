__all__ = [
    'NamedModulesFilter',
]

from typing import Generator

from torch import nn

from ..registries import FilterRegistry
from .named_member import NamedMembersFilter


@FilterRegistry.register_()
class NamedModulesFilter(NamedMembersFilter[nn.Module]):

    def _named_members(
        self,
        module: nn.Module,
    ) -> Generator[tuple[str, nn.Module], None, None]:
        return module.named_modules()
