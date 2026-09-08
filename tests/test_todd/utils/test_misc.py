import pytest
from custom_object import CustomObject

from todd.utils.misc import set_temp


def test_set_temp(custom_object: CustomObject) -> None:
    with set_temp(custom_object, '.one', 'I'):
        assert custom_object.one == 'I'
    assert custom_object.one == 1

    with pytest.raises(RuntimeError):
        with set_temp(custom_object, '.one', 'I'):
            raise RuntimeError
    assert custom_object.one == 1

    with set_temp(custom_object, '.zero', 0):
        assert custom_object.zero == 0
    assert not hasattr(custom_object, 'zero')

    with pytest.raises(RuntimeError):
        with set_temp(custom_object, '.zero', 0):
            raise RuntimeError
    assert not hasattr(custom_object, 'zero')
