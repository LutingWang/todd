import torch

from todd_torch.utils.statistician import Statistician


class TestStatistician:

    def test_update(self) -> None:
        statistician = Statistician(chunk_size=2)

        statistician.update(torch.tensor([[1.], [3.]]))

        assert statistician.num_samples == 2
        assert statistician.compute_mean().item() == 2

        statistician.update(torch.tensor([[5.]]))

        assert statistician.num_samples == 3
        assert statistician.compute_mean().item() == 3
