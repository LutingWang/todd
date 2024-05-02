import pytest
import torch

from todd.distillers.adapts.dict_tensor import Intersect


class TestIntersect:

    @pytest.fixture(scope='class')
    def intersect(self):
        return Intersect()

    def test_normal(self, intersect: Intersect):
        diff = 100
        feat1 = torch.arange(30).reshape(10, 3)
        feat2 = torch.arange(42).reshape(14, 3) + diff
        pos1 = torch.arange(10).unsqueeze(-1).repeat((1, 5))
        pos2 = torch.arange(14).unsqueeze(-1).repeat((1, 5))
        intersect_feats = intersect([feat1, feat2], [pos1, pos2])
        assert torch.all(intersect_feats[1] - intersect_feats[0] == 100)
