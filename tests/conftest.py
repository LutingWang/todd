import pytest

from todd.base import get_iter, init_iter, iter_initialized


@pytest.fixture
def setup_iter(setup_value):
    init_iter(setup_value)


@pytest.fixture
def setup_teardown_iter(setup_value, teardown_value):
    if teardown_value is ...:
        teardown_value = get_iter() if iter_initialized() else None

    if setup_value is not ...:
        init_iter(setup_value)

    yield
    init_iter(teardown_value)
