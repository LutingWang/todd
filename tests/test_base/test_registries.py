from abc import ABC, abstractmethod

import pytest

from todd.base import Registry


class TestRegistry:

    @pytest.fixture()
    def cats(self) -> Registry:
        return Registry('cats')

    @pytest.fixture()
    def dogs(self) -> Registry:
        return Registry('dogs')

    @pytest.mark.parametrize('registry', ['cats', 'dogs'])
    def test_init(self, registry: str, request: pytest.FixtureRequest):
        registry_: Registry = request.getfixturevalue(registry)
        assert registry_.name == registry
        assert registry_.parent is None
        assert registry_.modules == dict()
        assert registry_.children == dict()
        assert registry_.root == registry_

    def test_register_module_decorator(self, cats: Registry):
        assert len(cats) == 0

        @cats.register_module()
        class BritishShorthair:
            pass

        assert len(cats) == 1
        assert 'BritishShorthair' in cats
        assert cats.get('BritishShorthair') is BritishShorthair

        with pytest.raises(KeyError):

            @cats.register_module()
            class BritishShorthair:  # type: ignore[no-redef]
                pass

        assert len(cats) == 1
        assert 'BritishShorthair' in cats
        assert cats.get('BritishShorthair') is BritishShorthair

        @cats.register_module()
        class Munchkin:
            pass

        assert len(cats) == 2
        assert 'BritishShorthair' in cats
        assert 'Munchkin' in cats
        assert cats.get('BritishShorthair') is BritishShorthair
        assert cats.get('Munchkin') is Munchkin

        @cats.register_module(force=True)
        class Munchkin:  # type: ignore[no-redef]
            pass

        assert len(cats) == 2
        assert 'BritishShorthair' in cats
        assert 'Munchkin' in cats
        assert cats.get('BritishShorthair') is BritishShorthair
        assert cats.get('Munchkin') is Munchkin

        @cats.register_module()
        def munchkin():
            pass

        assert len(cats) == 3
        assert 'BritishShorthair' in cats
        assert 'Munchkin' in cats
        assert 'munchkin' in cats
        assert cats.get('BritishShorthair') is BritishShorthair
        assert cats.get('Munchkin') is Munchkin
        assert cats.get('munchkin') is munchkin

        assert 'PersianCat' not in cats
        with pytest.raises(KeyError):
            cats['PersianCat']
        assert cats.get('PersianCat') is None

        @cats.register_module(name='Siamese', aliases=('Siamese1', 'Siamese2'))
        class SiameseCat:
            pass

        assert len(cats) == 6
        assert 'SiameseCat' not in cats
        assert 'Siamese' in cats
        assert 'Siamese1' in cats
        assert 'Siamese2' in cats
        with pytest.raises(KeyError):
            cats['SiameseCat']
        assert cats['Siamese'].__name__ == 'SiameseCat'
        assert cats['Siamese1'].__name__ == 'SiameseCat'
        assert cats['Siamese2'].__name__ == 'SiameseCat'

        with pytest.raises(TypeError):

            @cats.register_module()
            class DomesticCat(ABC):

                @abstractmethod
                def meow(self) -> None:
                    pass

    def test_register_module_function(self, cats: Registry):
        assert len(cats) == 0

        class BritishShorthair:
            pass

        cats.register_module(BritishShorthair)
        assert len(cats) == 1
        assert 'BritishShorthair' in cats
        assert cats.get('BritishShorthair') is BritishShorthair

        with pytest.raises(KeyError):
            cats.register_module(BritishShorthair)
        assert len(cats) == 1
        assert 'BritishShorthair' in cats
        assert cats.get('BritishShorthair') is BritishShorthair

        class Munchkin:
            pass

        cats.register_module(Munchkin)
        assert len(cats) == 2
        assert 'BritishShorthair' in cats
        assert 'Munchkin' in cats
        assert cats.get('BritishShorthair') is BritishShorthair
        assert cats.get('Munchkin') is Munchkin

        cats.register_module(Munchkin, force=True)
        assert len(cats) == 2
        assert 'BritishShorthair' in cats
        assert 'Munchkin' in cats
        assert cats.get('BritishShorthair') is BritishShorthair
        assert cats.get('Munchkin') is Munchkin

        def munchkin():
            pass

        cats.register_module(munchkin)
        assert len(cats) == 3
        assert 'BritishShorthair' in cats
        assert 'Munchkin' in cats
        assert 'munchkin' in cats
        assert cats.get('BritishShorthair') is BritishShorthair
        assert cats.get('Munchkin') is Munchkin
        assert cats.get('munchkin') is munchkin

        assert 'PersianCat' not in cats
        with pytest.raises(KeyError):
            cats['PersianCat']
        assert cats.get('PersianCat') is None

        class SiameseCat:
            pass

        cats.register_module(
            SiameseCat,
            name='Siamese',
            aliases=('Siamese1', 'Siamese2'),
        )
        assert len(cats) == 6
        assert 'SiameseCat' not in cats
        assert 'Siamese' in cats
        assert 'Siamese2' in cats
        assert 'Siamese2' in cats
        with pytest.raises(KeyError):
            cats['SiameseCat']
        assert cats['Siamese'].__name__ == 'SiameseCat'
        assert cats['Siamese1'].__name__ == 'SiameseCat'
        assert cats['Siamese2'].__name__ == 'SiameseCat'

        class SphynxCat:
            pass

        cats.register_module(SphynxCat, name='Sphynx')
        assert len(cats) == 7
        assert 'Sphynx' in cats
        assert cats.get('Sphynx') is SphynxCat

        cats.register_module(SphynxCat, name='Sphynx1', aliases=['Sphynx2'])
        assert len(cats) == 9
        assert 'Sphynx1' in cats
        assert 'Sphynx2' in cats
        assert cats.get('Sphynx1') is SphynxCat
        assert cats.get('Sphynx2') is SphynxCat

        class DomesticCat(ABC):

            @abstractmethod
            def meow(self) -> None:
                pass

        with pytest.raises(TypeError):
            cats.register_module(DomesticCat)

    def test_inheritance(self, dogs: Registry):
        assert len(dogs) == 0

        @dogs.register_module()
        class GoldenRetriever:
            pass

        hounds = Registry('hounds', parent=dogs)
        assert dogs.children == dict(hounds=hounds)
        assert hounds.parent == dogs
        assert hounds.root == dogs

        assert dogs.get('GoldenRetriever') is GoldenRetriever
        assert hounds.get('GoldenRetriever') is None

        @hounds.register_module()
        class BloodHound:
            pass

        assert len(dogs) == 1
        assert len(hounds) == 1
        assert dogs.get('GoldenRetriever') is GoldenRetriever
        assert dogs.get('BloodHound') is None
        assert dogs.get('hounds.BloodHound') is BloodHound
        assert hounds.get('GoldenRetriever') is None
        assert hounds.get('BloodHound') is BloodHound
        assert hounds.get('hounds.BloodHound') is None

        little_hounds = Registry('little hounds', parent=hounds)
        assert dogs.children == dict(hounds=hounds)
        assert hounds.children == {'little hounds': little_hounds}
        assert hounds.parent == dogs
        assert little_hounds.parent == hounds
        assert hounds.root == dogs
        assert little_hounds.root == dogs

        @little_hounds.register_module()
        class Dachshund:
            pass

        assert dogs.get('Dachshund') is None
        assert dogs.get('hounds.Dachshund') is None
        assert dogs.get('little hounds.Dachshund') is None
        assert dogs.get('hounds.little hounds.Dachshund') is Dachshund
        assert hounds.get('Dachshund') is None
        assert hounds.get('hounds.Dachshund') is None
        assert hounds.get('little hounds.Dachshund') is Dachshund
        assert hounds.get('hounds.little hounds.Dachshund') is None

        mid_hounds = Registry('mid hounds', parent=hounds)
        assert dogs.children == dict(hounds=hounds)
        assert hounds.children == {
            'little hounds': little_hounds,
            'mid hounds': mid_hounds,
        }
        assert hounds.parent == dogs
        assert little_hounds.parent == hounds
        assert mid_hounds.parent == hounds
        assert hounds.root == dogs
        assert little_hounds.root == dogs
        assert mid_hounds.root == dogs

        @mid_hounds.register_module()
        class Beagle:
            pass

        assert dogs.get('hounds.mid hounds.Beagle') is Beagle
        assert hounds.get('mid hounds.Beagle') is Beagle
        assert mid_hounds.get('Beagle') is Beagle

        assert little_hounds.parent.get('mid hounds.Beagle') is Beagle
        assert little_hounds.root.get('hounds.mid hounds.Beagle') is Beagle

    def test_build(self):
        BACKBONES = Registry('backbone')

        @BACKBONES.register_module()
        class ResNet:

            def __init__(self, depth, stages=4):
                self.depth = depth
                self.stages = stages

        model = BACKBONES.build(dict(type='ResNet', depth=50))
        assert isinstance(model, ResNet)
        assert model.depth == 50 and model.stages == 4

        model = BACKBONES.build(dict(type='ResNet', depth=50, stages=3))
        assert isinstance(model, ResNet)
        assert model.depth == 50 and model.stages == 3

        model = BACKBONES.build(
            dict(type='ResNet', depth=50),
            default_args=dict(stages=3),
        )
        assert isinstance(model, ResNet)
        assert model.depth == 50 and model.stages == 3

        model = BACKBONES.build(
            dict(depth=50),
            default_args=dict(type='ResNet'),
        )
        assert isinstance(model, ResNet)
        assert model.depth == 50 and model.stages == 4

        model = BACKBONES.build(
            dict(depth=50),
            default_args=dict(type=ResNet),
        )
        assert isinstance(model, ResNet)
        assert model.depth == 50 and model.stages == 4

        with pytest.raises(TypeError, match='int'):
            BACKBONES.build(
                0,
                default_args=dict(type=ResNet, depth=50),
            )
        with pytest.raises(TypeError, match='int'):
            BACKBONES.build(
                dict(type=ResNet, depth=50),
                default_args=0,
            )

        with pytest.raises(KeyError):
            BACKBONES.build(dict(depth=50, stages=4))
        with pytest.raises(KeyError):
            BACKBONES.build(
                dict(depth=50),
                default_args=dict(stages=4),
            )
        with pytest.raises(KeyError):
            BACKBONES.build(dict(type='ResNeXt'))

        with pytest.raises(
            RuntimeError,
            match=(
                "Registry backbone failed to build ResNet with cfg "
                "\\{'non_existing_arg': 50\\}. TypeError: __init__\\(\\) got "
                "an unexpected keyword argument 'non_existing_arg'"
            ),
        ):
            BACKBONES.build(dict(type='ResNet', non_existing_arg=50))

    def test_build_base(self):

        class BaseBackbone:
            pass

        BACKBONES = Registry('backbone', base=BaseBackbone)

        model = BACKBONES.build(dict(type='BaseBackbone'))
        assert isinstance(model, BaseBackbone)

        @BACKBONES.register_module()
        class ResNet(BaseBackbone):

            def __init__(self, depth, stages=4):
                self.depth = depth
                self.stages = stages

        model = BACKBONES.build(dict(type='ResNet', depth=50))
        assert isinstance(model, ResNet)
        assert model.depth == 50 and model.stages == 4

        with pytest.raises(TypeError):
            BACKBONES.build(
                0,
                default_args=dict(type=ResNet, depth=50),
            )

        model = ResNet(depth=50)
        assert model is BACKBONES.build(model)

        model = ResNet(depth=50)
        with pytest.raises(ValueError):
            BACKBONES.build(
                model,
                default_args=dict(depth=50),
            )

        class BaseTransformer(ABC):

            @abstractmethod
            def forward(self):
                pass

        TRANSFORMER_BACKBONES = Registry(
            'transformer_backbone',
            base=BaseTransformer,
        )

        with pytest.raises(KeyError):
            TRANSFORMER_BACKBONES.build(dict(type='BaseTransformer'))

    def test_build_func(self):

        def build_func(registry: Registry, cfg: dict):
            type_ = registry[cfg.pop('class_')]
            return type_(depth=50, stages=3)  # type: ignore[operator]

        BACKBONES = Registry('backbone', build_func=build_func)

        @BACKBONES.register_module()
        class ResNet:

            def __init__(self, depth, stages=4):
                self.depth = depth
                self.stages = stages

        model = BACKBONES.build(dict(class_='ResNet'))
        assert isinstance(model, ResNet)
        assert model.depth == 50 and model.stages == 3

        def transform_build_func(registry: Registry, cfg: dict):
            type_ = registry[cfg.pop('transformer_class')]
            return type_(depth=100, stages=8)  # type: ignore[operator]

        TRANSFORMER_BACKBONES = Registry(
            'transformer_backbone',
            parent=BACKBONES,
            build_func=transform_build_func,
        )

        @TRANSFORMER_BACKBONES.register_module()
        class Transformer:

            def __init__(self, depth, stages=4):
                self.depth = depth
                self.stages = stages

        model = TRANSFORMER_BACKBONES.build(
            dict(transformer_class='Transformer'),
        )
        assert isinstance(model, Transformer)
        assert model.depth == 100 and model.stages == 8

    def test_build_errors(self):
        BACKBONES = Registry('backbone')

        @BACKBONES.register_module()
        class ResNet:

            def __init__(self, depth, stages=4):
                self.depth = depth
                self.stages = stages
