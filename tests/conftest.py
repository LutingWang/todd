import pytest

import todd.utils.iters as iters


@pytest.fixture
def setup_iter(setup_value):
    iters.init_iter(setup_value)


@pytest.fixture
def setup_teardown_iter(setup_value, teardown_value):
    if teardown_value is ...:
        teardown_value = iters.get_iter() if iters.iter_initialized() else None

    if setup_value is not ...:
        iters.init_iter(setup_value)

    yield
    iters.init_iter(teardown_value)
