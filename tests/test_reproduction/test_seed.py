import pytest
import torch
import torch.backends.cudnn as cudnn

from todd.base import init_iter
from todd.reproduction.seed import _randint, init_seed, set_seed_temp


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
    def seed534895718_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [18, 86, 21, 90, 49, 59, 20, 34, 90, 24],
            dtype=torch.int,
        )

    @pytest.fixture
    def seed198276314_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [35, 51, 98, 16, 62, 32, 77, 84, 77, 17],
            dtype=torch.int,
        )

    @pytest.mark.usefixtures('setup_teardown_iter_with_none')
    def test_randint(self) -> None:
        init_seed(42)
        assert _randint() == 534895718
        assert _randint() == 199900595

    @pytest.mark.usefixtures('setup_teardown_iter_with_none')
    def test_init_seed(
        self,
        seed42_tensor1: torch.Tensor,
        seed42_tensor2: torch.Tensor,
        seed534895718_tensor: torch.Tensor,
        seed198276314_tensor: torch.Tensor,
    ) -> None:
        init_seed(42)
        randint = torch.randint(0, 100, (10, ))
        assert randint.eq(seed42_tensor1).all()
        randint = torch.randint(0, 100, (10, ))
        assert randint.eq(seed42_tensor2).all()

        init_seed(42)
        init_seed()  # 534895718
        randint = torch.randint(0, 100, (10, ))
        assert randint.eq(seed534895718_tensor).all()

        init_seed('seed')
        randint = torch.randint(0, 100, (10, ))
        assert randint.eq(seed198276314_tensor).all()

        init_seed(b'seed')
        randint = torch.randint(0, 100, (10, ))
        assert randint.eq(seed198276314_tensor).all()

        init_seed(42, False)
        cudnn.deterministic = False
        cudnn.benchmark = True

        init_seed(42, True)
        cudnn.deterministic = True
        cudnn.benchmark = False

        init_iter(40)
        init_seed(2)
        randint = torch.randint(0, 100, (10, ))
        assert randint.eq(seed42_tensor1).all()

    @pytest.mark.usefixtures('setup_teardown_iter_with_none')
    def test_set_seed_temp(
        self,
        seed42_tensor1: torch.Tensor,
        seed42_tensor2: torch.Tensor,
        seed3407_tensor: torch.Tensor,
    ) -> None:
        init_seed(42)
        randint = torch.randint(0, 100, (10, ))
        assert randint.eq(seed42_tensor1).all()
        with set_seed_temp(3407):
            randint = torch.randint(0, 100, (10, ))
            assert randint.eq(seed3407_tensor).all()
        randint = torch.randint(0, 100, (10, ))
        assert randint.eq(seed42_tensor2).all()
