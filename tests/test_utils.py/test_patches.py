import pytest
from custom_types import CustomObject

from todd.utils.patches import del_, get_, has_, set_, set_temp


def test_has(obj: CustomObject) -> None:
    assert has_(obj, '.one')


def test_get(obj: CustomObject) -> None:
    assert get_(obj, '.one') == 1
    with pytest.raises(AttributeError, match='zero'):
        get_(obj, '.zero')
    assert get_(obj, '.zero', 0) == 0


def test_set(obj: CustomObject) -> None:
    set_(obj, '.obj.two', 2)
    assert obj.obj.two == 2

    with pytest.raises(ValueError, match='three'):
        set_(obj, 'three', 3)


def test_del(obj: CustomObject) -> None:
    with pytest.raises(AttributeError, match='zero'):
        del_(obj, '.zero')

    set_(obj, '.obj.two', 2)
    del_(obj, '.obj.two')
    assert not has_(obj, '.obj.two')


def test_set_temp(obj: CustomObject) -> None:
    with set_temp(obj, '.one', 'I'):
        assert obj.one == 'I'
    assert obj.one == 1

    with set_temp(obj, '.zero', 0):
        assert obj.zero == 0
    assert not hasattr(obj, 'zero')
