import pytest
from custom_object import CustomObject

from todd.patches.builtins import del_, get_, has_, set_


def test_has(custom_object: CustomObject) -> None:
    assert has_(custom_object, '.one')


def test_get(custom_object: CustomObject) -> None:
    assert get_(custom_object, '.one') == 1
    with pytest.raises(AttributeError, match='zero'):
        get_(custom_object, '.zero')
    assert get_(custom_object, '.zero', 0) == 0


def test_set(custom_object: CustomObject) -> None:
    set_(custom_object, '.obj.two', 2)
    assert custom_object.obj.two == 2

    with pytest.raises(ValueError, match='three'):
        set_(custom_object, 'three', 3)


def test_del(custom_object: CustomObject) -> None:
    with pytest.raises(AttributeError, match='zero'):
        del_(custom_object, '.zero')

    set_(custom_object, '.obj.two', 2)
    del_(custom_object, '.obj.two')
    assert not has_(custom_object, '.obj.two')
