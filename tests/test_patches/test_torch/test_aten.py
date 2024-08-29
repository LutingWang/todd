from todd.patches.torch.aten import random_int
from todd.utils import init_seed


def test_random_int() -> None:
    init_seed(42)
    assert random_int() == 534895718
    assert random_int() == 199900595
