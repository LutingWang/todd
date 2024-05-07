import pytest
import torch

from todd.reproducers.seed import init_seed, randint, set_seed_temp


class TestSeed:

    @pytest.fixture
    def seed42_tensor1(self) -> torch.Tensor:
        return torch.tensor(
            [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
            dtype=torch.int,
        )

    @pytest.fixture
    def seed42_tensor2(self) -> torch.Tensor:
        return torch.tensor(
            [78, 14, 10, 54, 31, 72, 15, 95, 67, 6],
            dtype=torch.int,
        )

    @pytest.fixture
    def seed3407_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [59, 68, 71, 49, 91, 10, 58, 80, 44, 86],
            dtype=torch.int,
        )

    @pytest.fixture
    def seed198276314_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [35, 51, 98, 16, 62, 32, 77, 84, 77, 17],
            dtype=torch.int,
        )

    def test_randint(self) -> None:
        init_seed(42)
        assert randint() == 534895718
        assert randint() == 199900595

    def test_init_seed(
        self,
        seed42_tensor1: torch.Tensor,
        seed42_tensor2: torch.Tensor,
    ) -> None:
        init_seed(42)
        randint_ = torch.randint(0, 100, (10, ))
        assert randint_.eq(seed42_tensor1).all()
        randint_ = torch.randint(0, 100, (10, ))
        assert randint_.eq(seed42_tensor2).all()

    def test_set_seed_temp(
        self,
        seed42_tensor1: torch.Tensor,
        seed42_tensor2: torch.Tensor,
        seed3407_tensor: torch.Tensor,
        seed198276314_tensor: torch.Tensor,
    ) -> None:
        init_seed(42)
        randint_ = torch.randint(0, 100, (10, ))
        assert randint_.eq(seed42_tensor1).all()
        with set_seed_temp(3407):
            randint_ = torch.randint(0, 100, (10, ))
            assert randint_.eq(seed3407_tensor).all()
        randint_ = torch.randint(0, 100, (10, ))
        assert randint_.eq(seed42_tensor2).all()

        with set_seed_temp('seed'):
            randint_ = torch.randint(0, 100, (10, ))
            assert randint_.eq(seed198276314_tensor).all()

        with set_seed_temp(b'seed'):
            randint_ = torch.randint(0, 100, (10, ))
            assert randint_.eq(seed198276314_tensor).all()
