import pytest

import todd.utils.iters as iters


@pytest.fixture
def setup_iter():
    iters.init_iter(None)


@pytest.fixture
def teardown_iter():
    yield
    iters.init_iter(None)


@pytest.fixture
def setup_teardown_iter():
    iters.init_iter(None)
    yield
    iters.init_iter(None)
