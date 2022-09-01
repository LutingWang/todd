import logging
import os
import pathlib
import sys
from typing import Generator, cast

import pytest
import torch.nn as nn

from todd.base import get_iter, init_iter, iter_initialized

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))
from custom_types import CustomModule, CustomObject  # noqa: E402


@pytest.fixture
def data_dir(request: pytest.FixtureRequest) -> pathlib.Path:
    return request.path.resolve().with_suffix('')


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


@pytest.fixture
def teardown_logger(logger_name: str) -> Generator[None, None, None]:
    yield
    logger = cast(
        logging.Logger,
        logging.Logger.manager.loggerDict.pop(logger_name),
    )
    for handler in logger.handlers:
        handler.close()


@pytest.fixture
def obj() -> CustomObject:
    return CustomObject(one=1, obj=CustomObject())


@pytest.fixture
def model() -> CustomModule:
    return CustomModule(
        conv=nn.Conv2d(128, 256, 3),
        module=CustomModule(linear=nn.Linear(1024, 10)),
    )
