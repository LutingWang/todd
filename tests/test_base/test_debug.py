import os

from todd.base.debug import DebugMode


class DebugEnum:
    CPU = DebugMode()


debug = DebugEnum()


class TestDebugMode:

    def test_get(self) -> None:
        assert not debug.CPU
        os.environ['CPU'] = '1'
        assert debug.CPU
        os.environ.pop('CPU')

    def test_set(self) -> None:
        debug.CPU = True
        assert 'CPU' in os.environ
        debug.CPU = False
        assert 'CPU' not in os.environ
