from typing import Generator

from torch import nn

from todd.base.filters import (
    NamedMembersFilter,
    NamedModulesFilter,
    NamedParametersFilter,
)


def named_members() -> list[tuple[str, int | str]]:
    return [('name1', 1), ('name2', '2'), ('name3', 3)]


class CustomNamedMembersFilter(NamedMembersFilter[int | str]):

    def _named_members(
        self,
        module: nn.Module,
    ) -> Generator[tuple[str, int | str], None, None]:
        yield from named_members()


class TestNamedMembersFilter:

    def test_filter_by_name(self) -> None:
        filter_ = CustomNamedMembersFilter(name='name1')
        result = filter_.filter_by_name(named_members())
        assert set(result) == {('name1', 1)}

        filter_ = CustomNamedMembersFilter(names=['name1', 'name2'])
        result = filter_.filter_by_name(named_members())
        assert set(result) == {('name1', 1), ('name2', '2')}

    def test_filter_by_regex(self) -> None:
        filter_ = CustomNamedMembersFilter(regex=r'^name[12]$')
        result = filter_.filter_by_regex(named_members())
        assert set(result) == {('name1', 1), ('name2', '2')}

    def test_filter_by_type(self) -> None:
        filter_ = CustomNamedMembersFilter(types=[int])
        result = filter_.filter_by_type(named_members())
        assert set(result) == {('name1', 1), ('name3', 3)}

    def test_call(self) -> None:
        module = nn.Module()

        filter_ = CustomNamedMembersFilter(name='name1', types=[int])
        result = filter_(module)
        assert set(result) == {('name1', 1)}

        filter_ = CustomNamedMembersFilter(name='name1', types=[str])
        result = filter_(module)
        assert not set(result)

        filter_ = CustomNamedMembersFilter(name='name1', types=[int, str])
        result = filter_(module)
        assert set(result) == {('name1', 1)}

        filter_ = CustomNamedMembersFilter(name='name1', regex=r'^name[12]$')
        result = filter_(module)
        assert set(result) == {('name1', 1)}

        filter_ = CustomNamedMembersFilter(
            name='name1',
            regex=r'^name[12]$',
            types=[int],
        )
        result = filter_(module)
        assert set(result) == {('name1', 1)}


class TestNamedModulesFilter:

    def test_named_members(self) -> None:
        filter_ = NamedModulesFilter()
        module = nn.Module()
        module.conv1 = nn.Conv2d(3, 64, 3)
        module.conv2 = nn.Conv2d(64, 64, 3)
        module.fc = nn.Linear(64, 10)
        result = filter_._named_members(module)
        assert set(result) == {
            ('', module),
            ('conv1', module.conv1),
            ('conv2', module.conv2),
            ('fc', module.fc),
        }


class TestNamedParametersFilter:

    def test_named_members(self) -> None:
        filter_ = NamedParametersFilter()
        module = nn.Module()
        module.conv1 = nn.Conv2d(3, 64, 3)
        module.conv2 = nn.Conv2d(64, 64, 3)
        module.fc = nn.Linear(64, 10)
        result = filter_._named_members(module)
        assert set(result) == {
            ('conv1.weight', module.conv1.weight),
            ('conv1.bias', module.conv1.bias),
            ('conv2.weight', module.conv2.weight),
            ('conv2.bias', module.conv2.bias),
            ('fc.weight', module.fc.weight),
            ('fc.bias', module.fc.bias),
        }
