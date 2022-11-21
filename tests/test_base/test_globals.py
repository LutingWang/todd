import unittest.mock as mock

import torch.cuda

from todd.base.globals_ import Device


class TestDevice:

    @mock.patch.object(torch.cuda, 'is_available', autospec=True)
    def test_cpu(self, mock_is_available: mock.MagicMock) -> None:
        mock_is_available.return_value = False

        device = Device()
        device.init()
        assert Device.CPU
        assert not Device.CUDA

        device.CPU = False
        assert not Device.CPU

    @mock.patch.object(torch.cuda, 'is_available', autospec=True)
    def test_cuda(self, mock_is_available: mock.MagicMock) -> None:
        mock_is_available.return_value = True

        device = Device()
        device.init()
        assert not device.CPU
        assert device.CUDA

        device.CUDA = False
        assert not Device.CUDA
