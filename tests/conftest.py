import os
import pathlib
import sys

import pytest
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))
from custom_types import CustomModule, CustomObject  # noqa: E402


@pytest.fixture
def data_dir(request: pytest.FixtureRequest) -> pathlib.Path:
    stem = request.path.stem.removeprefix('test_')
    return request.path.resolve().with_name(stem)


@pytest.fixture
def obj() -> CustomObject:
    return CustomObject(one=1, obj=CustomObject())


@pytest.fixture
def model() -> CustomModule:
    return CustomModule(
        conv=nn.Conv2d(128, 256, 3),
        module=CustomModule(linear=nn.Linear(1024, 10)),
    )
