from typing import Type, TypeVar

import pytest

from todd.base import Config, Registry

T = TypeVar('T')


class MyException(Exception):
    pass


@pytest.fixture
def cats() -> Registry:
    return Registry('cats')


class BritishShorthair:

    def __init__(self, name: str, age: int = 0):
        if age >= 20:
            raise MyException("Too old.")
        self._name = name
        self._age = age


class Munchkin:
    pass


class SphynxCat:
    pass


class Siamese:
    pass


@pytest.fixture
def dogs() -> Registry:
    return Registry('dogs')


@pytest.fixture
def hounds(dogs: Registry) -> Registry:
    return Registry('hounds', parent=dogs)


@pytest.fixture
def little_hounds(hounds: Registry) -> Registry:
    return Registry('little hounds', parent=hounds)


@pytest.fixture
def mid_hounds(hounds: Registry) -> Registry:
    return Registry('mid hounds', parent=hounds)


class BaseBackbone:

    def __init__(self, depth: int, stages: int = 4):
        self._depth = depth
        self._stages = stages


class ResNet(BaseBackbone):
    pass


class BaseMLP:
    pass


def build_func(registry: Registry, cfg: Config):
    type_: Type = registry[cfg.pop('class_')]
    return type_(depth=50, stages=3)  # type: ignore[operator]


class TestModules:

    def test_cats(self, cats: Registry):
        assert len(cats) == 0
        assert cats.modules == dict()

    def test_register(self, cats: Registry):
        cats.register(BritishShorthair)
        assert len(cats) == 1
        assert cats.modules == dict(BritishShorthair=BritishShorthair)
        assert 'BritishShorthair' in cats
        assert cats['BritishShorthair'] is BritishShorthair

        cats.register(Munchkin)
        assert len(cats) == 2
        assert cats.modules == dict(
            BritishShorthair=BritishShorthair,
            Munchkin=Munchkin,
        )
        assert 'Munchkin' in cats
        assert cats['Munchkin'] is Munchkin

        assert 'BritishShorthair' in cats
        assert cats['BritishShorthair'] is BritishShorthair

    def test_register_name(self, cats: Registry):
        cats.register(SphynxCat, name='Sphynx')
        assert len(cats) == 1
        assert 'Sphynx' in cats
        assert cats['Sphynx'] is SphynxCat
        assert cats.modules == dict(Sphynx=SphynxCat)

        assert 'SphynxCat' not in cats
        with pytest.raises(KeyError, match='SphynxCat'):
            cats['SphynxCat']

    def test_register_aliases(self, cats: Registry):
        cats.register(Siamese, aliases=('Siamese1', 'Siamese2'))
        assert len(cats) == 3
        assert 'Siamese' in cats
        assert 'Siamese1' in cats
        assert 'Siamese2' in cats
        assert cats['Siamese'] is Siamese
        assert cats['Siamese1'] is Siamese
        assert cats['Siamese2'] is Siamese
        assert cats.modules == dict(
            Siamese=Siamese,
            Siamese1=Siamese,
            Siamese2=Siamese,
        )

    def test_register_force(self, cats: Registry):
        cats.register(Munchkin)
        cats.register(Munchkin, force=True)
        assert len(cats) == 1
        assert 'Munchkin' in cats
        assert cats['Munchkin'] is Munchkin
        assert cats.modules == dict(Munchkin=Munchkin, )

    def test_register_errors(self, cats: Registry):
        cats.register(BritishShorthair)
        with pytest.raises(KeyError):
            cats.register(BritishShorthair)

        assert len(cats) == 1
        assert 'BritishShorthair' in cats
        assert cats['BritishShorthair'] is BritishShorthair
        assert cats.modules == dict(BritishShorthair=BritishShorthair, )

    def test_register_modules(self, cats: Registry):

        @cats.register_module()
        class BritishShorthair:
            pass

        assert cats.modules == dict(BritishShorthair=BritishShorthair)

    def test_get(self, cats: Registry):
        assert cats.get('non-exist') is None

        cats.register(BritishShorthair)
        assert cats.get('BritishShorthair') is BritishShorthair

    def test_build(self, cats: Registry):
        cats.register(BritishShorthair)

        cat = cats.build(dict(type='BritishShorthair', name='kitty'))
        assert isinstance(cat, BritishShorthair)
        assert cat._name == 'kitty'
        assert cat._age == 0

        cat = cats.build(dict(type='BritishShorthair', name='kitty', age=1))
        assert isinstance(cat, BritishShorthair)
        assert cat._name == 'kitty'
        assert cat._age == 1

        cat = cats.build(dict(type=BritishShorthair, name='kitty', age=2))
        assert isinstance(cat, BritishShorthair)
        assert cat._name == 'kitty'
        assert cat._age == 2

    def test_build_errors(self, cats: Registry):
        cats.register(BritishShorthair)
        with pytest.raises(KeyError, match='type'):
            cats.build(dict(name='kitty'))
        with pytest.raises(KeyError, match='non-exist'):
            cats.build(dict(type='non-exist'))
        with pytest.raises(MyException, match='Too old.'):
            cats.build(dict(type='BritishShorthair', name='kitty', age=20))


class TestParentMixin:

    def test_dogs(self, dogs: Registry):
        assert dogs.children == dict()
        with pytest.raises(AttributeError):
            dogs.parent
        assert dogs.root is dogs
        assert not dogs.has_parent()

    def test_hounds(self, dogs: Registry, hounds: Registry):
        assert dogs.children == dict(hounds=hounds)
        assert dogs.descendent('hounds') is hounds
        assert dogs.get_descendent('hounds') is hounds
        assert hounds.parent is dogs
        assert hounds.root is dogs
        assert hounds.has_parent()

    def test_little_hounds(
        self,
        dogs: Registry,
        hounds: Registry,
        little_hounds: Registry,
    ):
        assert dogs.children == dict(hounds=hounds)
        assert dogs.descendent('hounds.little hounds') is little_hounds
        assert dogs.get_descendent('hounds.little hounds') is little_hounds
        assert hounds.children == {'little hounds': little_hounds}
        assert hounds.descendent('little hounds') is little_hounds
        assert hounds.get_descendent('little hounds') is little_hounds
        assert little_hounds.parent is hounds
        assert little_hounds.root is dogs
        assert little_hounds.has_parent()

    def test_mid_hounds(
        self,
        dogs: Registry,
        hounds: Registry,
        little_hounds: Registry,
        mid_hounds: Registry,
    ):
        assert dogs.children == dict(hounds=hounds)
        assert dogs.descendent('hounds.mid hounds') is mid_hounds
        assert dogs.get_descendent('hounds.mid hounds') is mid_hounds
        assert hounds.children == {
            'little hounds': little_hounds,
            'mid hounds': mid_hounds,
        }
        assert hounds.descendent('mid hounds') is mid_hounds
        assert hounds.get_descendent('mid hounds') is mid_hounds
        assert mid_hounds.parent is hounds
        assert mid_hounds.root is dogs
        assert mid_hounds.has_parent()

    def test_descendent_errors(self, dogs: Registry):
        with pytest.raises(KeyError, match='non-exist'):
            dogs.descendent('non-exist')

    def test_get_descendent(self, dogs: Registry):
        assert dogs.get_descendent('non-exist') is None


class TestBaseMixin:

    def test_backbones(self):
        backbones = Registry('backbone', base=BaseBackbone)
        assert backbones.has_base()
        assert backbones.base is BaseBackbone

    def test_register_base(self):
        backbones = Registry('backbone', base=BaseBackbone, register_base=True)
        assert backbones.modules == dict(BaseBackbone=BaseBackbone)


class TestRegistry:

    def test_name(self, cats: Registry):
        assert cats.name == 'cats'

    def test_parent_base(self):
        backbones = Registry('backbone', base=BaseBackbone)
        conv_backbones = Registry('conv backbone', parent=backbones)
        assert conv_backbones.has_base()
        assert conv_backbones.base is BaseBackbone

        with pytest.raises(TypeError):
            Registry(
                'mlp backbones',
                parent=backbones,
                base=BaseMLP,
            )

    def test_parent_build_func(self):
        backbones = Registry('backbone', build_func=build_func)
        conv_backbones = Registry('conv backbone', parent=backbones)
        assert conv_backbones.build_func is build_func

    def test_register(self):
        backbones = Registry('backbones', base=BaseBackbone)
        with pytest.raises(TypeError, match='BaseBackbone'):
            backbones.register(BaseMLP)

    def test_get(
        self,
        dogs: Registry,
        hounds: Registry,
        little_hounds: Registry,
        mid_hounds: Registry,
    ):

        @dogs.register_module()
        class GoldenRetriever:
            pass

        @hounds.register_module()
        class BloodHound:
            pass

        @little_hounds.register_module()
        class Dachshund:
            pass

        @mid_hounds.register_module()
        class Beagle:
            pass

        assert dogs.get('GoldenRetriever') is GoldenRetriever
        assert hounds.get('GoldenRetriever') is None

        assert dogs.get('BloodHound') is None
        assert dogs.get('hounds.BloodHound') is BloodHound
        assert hounds.get('BloodHound') is BloodHound

        assert dogs.get('hounds.little hounds.Dachshund') is Dachshund
        assert dogs.get('hounds.mid hounds.Beagle') is Beagle

        assert dogs.get('hounds.Dachshund') is None
        assert dogs.get('little hounds.Dachshund') is None
        assert dogs.get('little hounds.hounds.Dachshund') is None

    def test_build(self):
        backbones = Registry('backbones')
        backbones.register(ResNet)

        resnet = ResNet(depth=101)
        model = backbones.build(resnet)
        assert model is resnet

    def test_build_base(self):
        backbones = Registry('backbones', base=BaseBackbone)
        backbones.register(ResNet)

        model = backbones.build(
            dict(type='ResNet', depth=50),
            default_args=dict(stages=3),
        )
        assert isinstance(model, ResNet)
        assert model._depth == 50 and model._stages == 3

        resnet = ResNet(depth=101)
        model = backbones.build(resnet)
        assert model is resnet

    def test_build_base_errors(self):
        backbones = Registry('backbones', base=BaseBackbone)
        with pytest.raises(TypeError, match='BaseMLP'):
            backbones.build(BaseMLP())

    def test_build_func(self):
        backbones = Registry('backbones', build_func=build_func)
        backbones.register(ResNet)

        model = backbones.build(dict(class_='ResNet', depth=50))
        assert isinstance(model, ResNet)
        assert model._depth == 50 and model._stages == 3
