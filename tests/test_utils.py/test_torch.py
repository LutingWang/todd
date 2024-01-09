import torch
from torch import nn

from todd.utils.torch import Shape


class TestShapePytest:

    def test_conv(self):
        module = nn.Conv1d(3, 6, 3, padding=1)
        x = torch.randn(2, 3, 4)
        assert Shape.conv(module, x) == (2, 6, 4)

        module = nn.Conv2d(3, 6, 3, padding=1)
        x = torch.randn(2, 3, 4, 4)
        assert Shape.conv(module, x) == (2, 6, 4, 4)

        module = nn.Conv2d(1, 1, 3, padding=1)
        x = torch.randn(1, 1, 5, 5)
        assert Shape.conv(module, x) == (1, 1, 5, 5)

        module = nn.Conv2d(3, 2, 3, padding=1)
        x = torch.randn(1, 3, 6, 6)
        assert Shape.conv(module, x) == (1, 2, 6, 6)

        module = nn.Conv2d(1, 1, 1)
        x = torch.randn(1, 1, 10, 10)
        assert Shape.conv(module, x) == (1, 1, 10, 10)
