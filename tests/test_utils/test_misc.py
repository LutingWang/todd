from custom_types import CustomObject

from todd.utils.misc import set_temp


def test_set_temp(obj: CustomObject) -> None:
    with set_temp(obj, '.one', 'I'):
        assert obj.one == 'I'
    assert obj.one == 1

    with set_temp(obj, '.zero', 0):
        assert obj.zero == 0
    assert not hasattr(obj, 'zero')
