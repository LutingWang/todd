import os
import sys
from typing import Generator

import pytest
import torch.nn as nn

from todd.base import get_iter, init_iter, iter_initialized

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))


@pytest.fixture
def setup_iter(setup_value) -> None:
    init_iter(setup_value)


@pytest.fixture
def setup_teardown_iter(
    setup_value,
    teardown_value,
) -> Generator[None, None, None]:
    if teardown_value is ...:
        teardown_value = get_iter() if iter_initialized() else None
    if setup_value is not ...:
        init_iter(setup_value)
    yield
    init_iter(teardown_value)


@pytest.fixture
def setup_teardown_iter_with_none() -> Generator[None, None, None]:
    init_iter(None)
    yield
    init_iter(None)


class CustomObject:

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def obj() -> CustomObject:
    return CustomObject(one=1, obj=CustomObject())


@pytest.fixture
def model() -> nn.Module:
    module = nn.Module()
    module.conv = nn.Conv2d(128, 256, 3)
    module.module = nn.Module()
    module.module.linear = nn.Linear(1024, 10)
    return module
