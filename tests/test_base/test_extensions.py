import pytest
from custom_types import CustomObject

from todd.base._extensions.builtins import (
    delattr_recur,
    getattr_recur,
    hasattr_recur,
    setattr_recur,
    setattr_temp,
)


def test_hasattr(obj: CustomObject) -> None:
    assert hasattr_recur(obj, 'one')
    assert hasattr_recur(obj, '.one')


def test_getattr(obj: CustomObject) -> None:
    assert getattr_recur(obj, 'one') == 1
    assert getattr_recur(obj, '.one') == 1
    with pytest.raises(AttributeError, match='zero'):
        getattr_recur(obj, 'zero')
    with pytest.raises(AttributeError, match='zero'):
        getattr_recur(obj, '.zero')
    assert getattr_recur(obj, 'zero', 0) == 0
    assert getattr_recur(obj, '.zero', 0) == 0


def test_setattr_recur(obj: CustomObject) -> None:
    setattr_recur(obj, 'one', 'I')
    assert getattr_recur(obj, 'one') == 'I'

    setattr_recur(obj, '.one', 'i')
    assert getattr_recur(obj, 'one') == 'i'

    setattr_recur(obj, 'zero', 0)
    assert getattr_recur(obj, 'zero') == 0

    setattr_recur(obj, '.two', 2)
    assert getattr_recur(obj, 'two') == 2

    setattr_recur(obj, '.obj.two', 2)
    assert getattr_recur(obj, '.obj.two') == 2


def test_delattr(obj: CustomObject) -> None:
    with pytest.raises(AttributeError, match='zero'):
        delattr_recur(obj, 'zero')
    with pytest.raises(AttributeError, match='zero'):
        delattr_recur(obj, '.zero')

    setattr_recur(obj, 'zero', 0)
    delattr_recur(obj, 'zero')
    assert not hasattr_recur(obj, 'zero')

    setattr_recur(obj, 'zero', 0)
    delattr_recur(obj, '.zero')
    assert not hasattr_recur(obj, 'zero')

    setattr_recur(obj, '.obj.two', 2)
    delattr_recur(obj, '.obj.two')
    assert not hasattr_recur(obj, '.obj.two')


def test_setattr_temp(obj: CustomObject) -> None:
    with setattr_temp(obj, 'one', 'I'):
        assert obj.one == 'I'
    assert obj.one == 1

    with setattr_temp(obj, '.one', 'I'):
        assert obj.one == 'I'
    assert obj.one == 1

    with setattr_temp(obj, 'zero', 0):
        assert obj.zero == 0
    assert not hasattr(obj, 'zero')

    with setattr_temp(obj, '.zero', 0):
        assert obj.zero == 0
    assert not hasattr(obj, 'zero')
