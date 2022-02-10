from typing import Any

import pytest
import torch

from todd.adapts import IoU


class TestIoU:
    @pytest.fixture(scope='class')
    def iou(self):
        return IoU()

    def test_normal(self, iou: IoU):
        bboxes1 = torch.Tensor([[
            [10, 10, 20, 20],
        ]])
        bboxes2 = torch.Tensor([[
            [11, 12, 19, 21],
            [5, 9, 25, 15],
        ]])
        result = torch.HalfTensor([[
            [64 / 108, 5 / 17],
        ]])
        assert torch.allclose(result, iou(bboxes1, bboxes2))
