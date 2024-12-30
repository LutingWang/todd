__all__ = [
    'NamedMembersFilter',
]

import re
from abc import abstractmethod
from typing import Generator, Iterable, TypeVar

from torch import nn

from .base import BaseFilter

T = TypeVar('T')


class NamedMembersFilter(BaseFilter[tuple[str, T]]):

    def __init__(
        self,
        *args,
        name: str | None = None,
        names: Iterable[str] | None = None,
        regex: str | None = None,
        type_: type[T] | None = None,
        types: Iterable[type[T]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert name is None or names is None
        if name is not None:
            names = (name, )

        assert type_ is None or types is None
        if type_ is not None:
            types = (type_, )

        self._names = None if names is None else tuple(names)
        self._regex = None if regex is None else re.compile(regex)
        self._types = None if types is None else tuple(types)

    @property
    def names(self) -> tuple[str, ...]:
        assert self._names is not None
        return self._names

    @property
    def regex(self) -> re.Pattern[str]:
        assert self._regex is not None
        return self._regex

    @property
    def types(self) -> tuple[type[T], ...]:
        assert self._types is not None
        return self._types

    def filter_by_name(
        self,
        named_members: Iterable[tuple[str, T]],
    ) -> Generator[tuple[str, T], None, None]:
        for name, member in named_members:
            if name in self.names:
                yield name, member

    def filter_by_regex(
        self,
        named_members: Iterable[tuple[str, T]],
    ) -> Generator[tuple[str, T], None, None]:
        for name, member in named_members:
            if self.regex.match(name):
                yield name, member

    def filter_by_type(
        self,
        named_members: Iterable[tuple[str, T]],
    ) -> Generator[tuple[str, T], None, None]:
        for name, member in named_members:
            if isinstance(member, self.types):
                yield name, member

    @abstractmethod
    def _named_members(
        self,
        module: nn.Module,
    ) -> Generator[tuple[str, T], None, None]:
        pass

    def __call__(
        self,
        module: nn.Module,
    ) -> Generator[tuple[str, T], None, None]:
        named_members = self._named_members(module)
        if self._names is not None:
            named_members = self.filter_by_name(named_members)
        if self._regex is not None:
            named_members = self.filter_by_regex(named_members)
        if self._types is not None:
            named_members = self.filter_by_type(named_members)
        yield from named_members
