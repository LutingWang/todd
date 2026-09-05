import pytest
import torch

from todd_torch.utils.seeds import init_seed, set_seed_temp

SEED42_TENSOR = torch.tensor(
    [42, 67, 76, 14, 26, 35, 20, 24, 50, 13],
    dtype=torch.long,
)


def test_init_seed() -> None:
    init_seed(42)
    assert torch.allclose(torch.randint(0, 100, (10, )), SEED42_TENSOR)


def test_set_seed_temp() -> None:
    init_seed(3407)
    expected_3407 = torch.rand(1)

    # NOTE: 198276314 is the encoded integer seed for b'seed'.
    init_seed(198276314)
    expected_seed = torch.rand(1)

    init_seed(42)
    with set_seed_temp(3407):
        assert torch.allclose(torch.rand(1), expected_3407)
    assert torch.allclose(torch.randint(0, 100, (10, )), SEED42_TENSOR)

    init_seed(42)
    with set_seed_temp('seed'):
        assert torch.allclose(torch.rand(1), expected_seed)
    assert torch.allclose(torch.randint(0, 100, (10, )), SEED42_TENSOR)

    init_seed(42)
    with set_seed_temp(b'seed'):
        assert torch.allclose(torch.rand(1), expected_seed)
    assert torch.allclose(torch.randint(0, 100, (10, )), SEED42_TENSOR)

    init_seed(42)
    with pytest.raises(RuntimeError):
        with set_seed_temp(3407):
            torch.rand(1)
            raise RuntimeError
    assert torch.allclose(torch.randint(0, 100, (10, )), SEED42_TENSOR)
