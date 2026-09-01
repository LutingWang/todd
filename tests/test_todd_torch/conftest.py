import pytest
from custom_module import CustomModule
from torch import nn


@pytest.fixture
def model() -> CustomModule:
    return CustomModule(
        conv=nn.Conv2d(128, 256, 3),
        module=CustomModule(linear=nn.Linear(1024, 10)),
    )
