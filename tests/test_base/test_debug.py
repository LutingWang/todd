import os
import unittest.mock as mock

import torch.cuda

from todd.base.debug import BaseDebug, DebugMode


class Debug(BaseDebug):
    CUDA = DebugMode()
    ASYNC_BATCH_NORM = DebugMode()
    DRY_RUN = DebugMode()

    def init_cuda(self, **kwargs) -> None:
        super().init_cuda(**kwargs)
        self.CUDA = True

    def init_cpu(self, **kwargs) -> None:
        super().init_cpu(**kwargs)
        self.ASYNC_BATCH_NORM = True

    def init_custom(self, **kwargs) -> None:
        super().init_custom(**kwargs)
        self.DRY_RUN = True


debug = Debug()


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


class TestDebug:

    def test_cpu(self) -> None:
        debug.init()
        assert debug.CPU
        assert not debug.CUDA
        assert debug.ASYNC_BATCH_NORM
        assert debug.DRY_RUN

        debug.CPU = False
        debug.ASYNC_BATCH_NORM = False
        debug.DRY_RUN = False

    @mock.patch.object(torch.cuda, 'is_available', autospec=True)
    def test_cuda(self, mock_is_available: mock.MagicMock) -> None:
        mock_is_available.return_value = True
        debug.init()
        assert not debug.CPU
        assert debug.CUDA
        assert not debug.ASYNC_BATCH_NORM
        assert debug.DRY_RUN

        debug.CUDA = False
        debug.DRY_RUN = False
