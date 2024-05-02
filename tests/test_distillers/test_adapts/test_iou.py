import pytest
import torch

from todd.distillers.adapts import IoU


class TestIoU:

    @pytest.fixture(scope='class')
    def iou(self):
        return IoU()

    def test_normal(self, iou: IoU):
        bboxes1 = [torch.Tensor([[
            [10, 10, 20, 20],
        ]])]
        bboxes2 = [torch.Tensor([[
            [11, 12, 19, 21],
            [5, 9, 25, 15],
        ]])]
        result = torch.Tensor([[
            [64 / 108, 5 / 17],
        ]])
        ious = iou(bboxes1, bboxes2)
        assert len(ious) == 1
        ious = ious[0]
        assert len(ious) == 1
        ious = ious[0]
        assert torch.allclose(result, ious)
