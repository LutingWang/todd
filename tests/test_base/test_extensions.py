from todd.base._extensions.builtins import setattr_temp


def test_setattr_temp(obj) -> None:
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
