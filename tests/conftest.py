import os
import pathlib
import sys

import pytest
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))
from custom_types import CustomModule, CustomObject  # noqa: E402


@pytest.fixture
def data_dir(request: pytest.FixtureRequest) -> pathlib.Path:
    return request.path.resolve().with_suffix('')


@pytest.fixture
def obj() -> CustomObject:
    return CustomObject(one=1, obj=CustomObject())


@pytest.fixture
def model() -> CustomModule:
    return CustomModule(
        conv=nn.Conv2d(128, 256, 3),
        module=CustomModule(linear=nn.Linear(1024, 10)),
    )
