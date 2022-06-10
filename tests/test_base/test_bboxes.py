import numpy as np
import pytest
import torch

from todd.base.bboxes import BBoxesXYXY


class TestBBoxesXYXY:

    @pytest.fixture(scope='class')
    def bboxes_np(self) -> np.ndarray:
        return np.array([
            [0.0, 10.0, 20.0, 40.0],
        ])

    @pytest.fixture(scope='class')
    def bboxes_torch(self) -> torch.Tensor:
        return torch.tensor([
            [0.0, 10.0, 20.0, 40.0],
        ])

    @pytest.mark.parametrize(
        'bboxes',
        ['bboxes_np', 'bboxes_torch'],
    )
    def test_basics(self, bboxes: str, request: pytest.FixtureRequest) -> None:
        bboxes_ = BBoxesXYXY(request.getfixturevalue(bboxes))
        assert len(bboxes_) == 1
        assert torch.tensor([0.0]).eq(bboxes_.left).all()
        assert torch.tensor([20.0]).eq(bboxes_.right).all()
        assert torch.tensor([10.0]).eq(bboxes_.top).all()
        assert torch.tensor([40.0]).eq(bboxes_.bottom).all()
        assert torch.tensor([20.0]).eq(bboxes_.width).all()
        assert torch.tensor([30.0]).eq(bboxes_.height).all()
        assert torch.tensor([10.0]).eq(bboxes_.center_x).all()
        assert torch.tensor([25.0]).eq(bboxes_.center_y).all()
        assert torch.tensor([0.0, 10.0]).eq(bboxes_.lt).all()
        assert torch.tensor([20.0, 40.0]).eq(bboxes_.rb).all()
        assert torch.tensor([20.0, 30.0]).eq(bboxes_.wh).all()
        assert torch.tensor([10.0, 25.0]).eq(bboxes_.center).all()
        assert (
            torch.tensor([0.0, 10.0, 20.0, 40.0]).eq(bboxes_.to_tensor()).all()
        )
        assert not bboxes_.empty()
        assert torch.tensor([20.0, 30.0]).eq(bboxes_.shapes).all()
        assert torch.tensor([600.0]).eq(bboxes_.areas).all()
