import os
import sys

import pytest

from todd.base import get_iter, init_iter, iter_initialized

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))


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


class CustomObject:

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture()
def obj() -> CustomObject:
    return CustomObject(one=1, obj=CustomObject())
