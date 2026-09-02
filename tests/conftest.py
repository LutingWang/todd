import pathlib

import pytest
from custom_object import (  # noqa: E402 pylint: disable=wrong-import-position
    CustomObject,
)


@pytest.fixture
def data_dir(request: pytest.FixtureRequest) -> pathlib.Path:
    stem = request.path.stem.removeprefix('test_')
    return request.path.resolve().with_name(stem)


@pytest.fixture
def obj() -> CustomObject:
    return CustomObject(one=1, obj=CustomObject())
