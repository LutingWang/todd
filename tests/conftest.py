import pathlib

import pytest
from custom_object import CustomObject


@pytest.fixture
def data_dir(request: pytest.FixtureRequest) -> pathlib.Path:
    stem = request.path.stem.removeprefix('test_')
    return request.path.resolve().with_name(stem)


@pytest.fixture
def custom_object() -> CustomObject:
    return CustomObject(one=1, obj=CustomObject())
