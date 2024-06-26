from typing import Any

from todd import Config
from todd.bases.registries import BaseBuilder, Builder


class ConcreteBuilder(BaseBuilder):

    def should_build(self, obj: Any) -> bool:
        raise NotImplementedError

    def build(self, obj: Any, **kwargs) -> Any:
        raise NotImplementedError


def f(*args, **kwargs) -> Any:
    raise NotImplementedError


class TestBaseBuilder:

    def test_requires(self) -> None:
        builder = ConcreteBuilder(
            f,
            requires={
                'arg1': 'key1',
                'arg2': 'key2',
            },
        )
        assert builder.requires == ('arg1', 'arg2')


class TestBuilder:

    def test_should_build(self) -> None:
        builder = Builder(f)
        assert builder.should_build(Config())
        assert not builder.should_build('not a Config object')

    def test_build(self) -> None:
        builder = Builder(lambda obj, x, y: obj.value + x + y)
        obj = Config(value=1)
        assert builder.build(obj, x=2, y=3) == 6
