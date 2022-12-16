import logging
import os
import pathlib
import sys
from typing import Generator, cast

import pytest
import torch.nn as nn

from todd import Store

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))
from custom_types import CustomModule, CustomObject  # noqa: E402


@pytest.fixture
def data_dir(request: pytest.FixtureRequest) -> pathlib.Path:
    return request.path.resolve().with_suffix('')


@pytest.fixture
def setup_teardown_iter(
    setup_value: int = 0,
    teardown_value: int = 0,
) -> Generator[None, None, None]:
    Store.ITER = setup_value
    yield
    Store.ITER = teardown_value


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
