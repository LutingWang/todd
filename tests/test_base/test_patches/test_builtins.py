import pytest


class CustomObject:

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture()
def obj() -> CustomObject:
    return CustomObject(one=1)


def test_hasattr(obj: CustomObject) -> None:
    assert hasattr(obj, 'one')
    assert hasattr(obj, '.one')


def test_getattr(obj: CustomObject) -> None:
    assert getattr(obj, 'one') == 1
    assert getattr(obj, '.one') == 1
    with pytest.raises(AttributeError, match='zero'):
        getattr(obj, 'zero')
    with pytest.raises(AttributeError, match='zero'):
        getattr(obj, '.zero')
    assert getattr(obj, 'zero', 0) == 0
    assert getattr(obj, '.zero', 0) == 0


def test_setattr(obj: CustomObject) -> None:
    setattr(obj, 'one', 'I')
    assert getattr(obj, 'one') == 'I'

    setattr(obj, '.one', 'i')
    assert getattr(obj, 'one') == 'i'

    setattr(obj, 'zero', 0)
    assert getattr(obj, 'zero') == 0

    setattr(obj, '.two', 2)
    assert getattr(obj, 'two') == 2


def test_delattr(obj: CustomObject) -> None:
    with pytest.raises(AttributeError, match='zero'):
        delattr(obj, 'zero')
    with pytest.raises(AttributeError, match='zero'):
        delattr(obj, '.zero')

    setattr(obj, 'zero', 0)
    delattr(obj, 'zero')
    assert not hasattr(obj, 'zero')

    setattr(obj, 'zero', 0)
    delattr(obj, '.zero')
    assert not hasattr(obj, 'zero')
