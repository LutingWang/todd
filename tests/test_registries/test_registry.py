import pytest

from todd import Config
from todd.registries.registry import Registry, RegistryMeta


class Registry1(metaclass=RegistryMeta):
    pass


class Registry2(Registry1):
    pass


class Registry3(Registry2):
    pass


class Registry3_1(Registry2):  # noqa: N801 pylint: disable=invalid-name
    pass


class TestRegistryMeta:

    def test_call(self) -> None:
        with pytest.raises(RuntimeError):
            Registry()

    def test_missing(self) -> None:
        key = 'custom_key'
        with pytest.raises(KeyError, match=key):
            Registry[key]  # pylint: disable=pointless-statement

    def test_parse(self) -> None:
        registry, key = Registry1._parse('Registry2.Registry3.custom_key')
        assert registry is Registry3
        assert key == 'custom_key'

    def test_child(self) -> None:
        with pytest.raises(ValueError):
            Registry1.child('Registry1')
        with pytest.raises(ValueError):
            Registry1.child('Registry2.Registry2')

        class Registry3_1(  # noqa: E501,N801 pylint: disable=invalid-name,redefined-outer-name,unused-variable
            Registry2,
        ):
            pass

        with pytest.raises(ValueError):
            Registry1.child('Registry2.Registry3_1')

    def test_build(self) -> None:
        with pytest.raises(KeyError):
            Registry.build(Config(type='custom_key'))
