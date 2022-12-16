# import unittest.mock as mock

# import torch.cuda

# from todd.utils.trials import trial_store

# class TestDevice:

#     @mock.patch.object(torch.cuda, 'is_available', autospec=True)
#     def test_cpu(self, mock_is_available: mock.MagicMock) -> None:
#         mock_is_available.return_value = False

#         assert trial_store.CPU
#         assert not trial_store.CUDA

#         trial_store.CPU = False
#         assert not trial_store.CPU

#     @mock.patch.object(torch.cuda, 'is_available', autospec=True)
#     def test_cuda(self, mock_is_available: mock.MagicMock) -> None:
#         mock_is_available.return_value = True

#         assert not trial_store.CPU
#         assert trial_store.CUDA

#         trial_store.CUDA = False
#         assert not trial_store.CUDA
