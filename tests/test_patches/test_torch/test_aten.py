from todd.patches.torch.aten import random_int
from todd.utils import init_seed


def test_random_int() -> None:
    init_seed(42)
    assert random_int() == 199900595
    assert random_int() == 787846414
