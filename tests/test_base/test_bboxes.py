import numpy as np
import pytest
import torch

from todd.base.bboxes import BBoxesXY, BBoxesXYWH, BBoxesXYXY


class TestBBoxes:

    def test_init(self):
        bboxes_np = np.array([[10.0, 20.0, 40.0, 100.0]])
        bboxes_torch = torch.tensor([[10.0, 20.0, 40.0, 100.0]])

        bboxes = BBoxesXYXY(bboxes_np)
        assert bboxes.to_tensor().eq(bboxes_torch).all()

        bboxes = BBoxesXYXY(bboxes_torch)
        assert bboxes.to_tensor().eq(bboxes_torch).all()

        bboxes = BBoxesXYXY([])
        assert bboxes.to_tensor().shape == (0, 4)

        bboxes = BBoxesXYXY(bboxes_torch.tolist())
        assert bboxes.to_tensor().eq(bboxes_torch).all()

        with pytest.raises(ValueError):
            BBoxesXYXY(bboxes_np[0])

        with pytest.raises(ValueError):
            BBoxesXYXY(bboxes_np[:, :3])

    def test_operators(self):
        # yapf: disable
        a = torch.tensor(
            [[ 10.0,  20.0,  40.0, 100.0]],
        )
        b = torch.tensor(
            [[  5.0,  15.0,   8.0,  18.0],
             [  5.0,  15.0,   8.0,  60.0],
             [  5.0,  15.0,   8.0, 105.0],
             [  5.0,  60.0,   8.0,  80.0],
             [  5.0,  60.0,   8.0, 105.0],
             [  5.0, 102.0,   8.0, 105.0],
             [  5.0,  15.0,  25.0,  18.0],
             [  5.0,  15.0,  25.0,  60.0],
             [  5.0,  15.0,  25.0, 105.0],
             [  5.0,  60.0,  25.0,  80.0],
             [  5.0,  60.0,  25.0, 105.0],
             [  5.0, 102.0,  25.0, 105.0],
             [  5.0,  15.0,  45.0,  18.0],
             [  5.0,  15.0,  45.0,  60.0],
             [  5.0,  15.0,  45.0, 105.0],
             [  5.0,  60.0,  45.0,  80.0],
             [  5.0,  60.0,  45.0, 105.0],
             [  5.0, 102.0,  45.0, 105.0],
             [ 25.0,  15.0,  35.0,  18.0],
             [ 25.0,  15.0,  35.0,  60.0],
             [ 25.0,  15.0,  35.0, 105.0],
             [ 25.0,  60.0,  35.0,  80.0],
             [ 25.0,  60.0,  35.0, 105.0],
             [ 25.0, 102.0,  35.0, 105.0],
             [ 25.0,  15.0,  45.0,  18.0],
             [ 25.0,  15.0,  45.0,  60.0],
             [ 25.0,  15.0,  45.0, 105.0],
             [ 25.0,  60.0,  45.0,  80.0],
             [ 25.0,  60.0,  45.0, 105.0],
             [ 25.0, 102.0,  45.0, 105.0],
             [ 42.0,  15.0,  45.0,  18.0],
             [ 42.0,  15.0,  45.0,  60.0],
             [ 42.0,  15.0,  45.0, 105.0],
             [ 42.0,  60.0,  45.0,  80.0],
             [ 42.0,  60.0,  45.0, 105.0],
             [ 42.0, 102.0,  45.0, 105.0]],
        )

        bboxes_a = BBoxesXYXY(a)
        bboxes_b = BBoxesXYXY(b)

        assert len(bboxes_a) == 1
        assert len(bboxes_b) == 36

        cat = torch.cat([a, b])
        bboxes_cat = bboxes_a.cat(bboxes_b)
        assert isinstance(bboxes_cat, BBoxesXYXY)
        assert len(bboxes_cat) == 37
        assert bboxes_cat.to_tensor().eq(cat).all()
        bboxes_cat = bboxes_a + bboxes_b
        assert isinstance(bboxes_cat, BBoxesXYXY)
        assert len(bboxes_cat) == 37
        assert bboxes_cat.to_tensor().eq(cat).all()

        selected = b[[10]]
        bboxes_selected = bboxes_b.select([10])
        assert isinstance(bboxes_selected, BBoxesXYXY)
        assert len(bboxes_selected) == 1
        assert bboxes_selected.to_tensor().eq(selected).all()
        bboxes_selected = bboxes_b[[10]]
        assert isinstance(bboxes_selected, BBoxesXYXY)
        assert len(bboxes_selected) == 1
        assert bboxes_selected.to_tensor().eq(selected).all()

        intersections = torch.tensor([[
               0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
               0.0,  600.0, 1200.0,  300.0,  600.0,    0.0,
               0.0, 1200.0, 2400.0,  600.0, 1200.0,    0.0,
               0.0,  400.0,  800.0,  200.0,  400.0,    0.0,
               0.0,  600.0, 1200.0,  300.0,  600.0,    0.0,
               0.0,    0.0,    0.0,    0.0,    0.0,    0.0,
        ]])
        assert bboxes_a.intersections(bboxes_b).eq(intersections).all()
        assert intersections.eq(bboxes_a & bboxes_b).all()

        unions = torch.tensor([[
            2409.0, 2535.0, 2670.0, 2460.0, 2535.0, 2409.0,
            2460.0, 2700.0, 3000.0, 2500.0, 2700.0, 2460.0,
            2520.0, 3000.0, 3600.0, 2600.0, 3000.0, 2520.0,
            2430.0, 2450.0, 2500.0, 2400.0, 2450.0, 2430.0,
            2460.0, 2700.0, 3000.0, 2500.0, 2700.0, 2460.0,
            2409.0, 2535.0, 2670.0, 2460.0, 2535.0, 2409.0,
        ]])
        assert bboxes_a.unions(bboxes_b).eq(unions).all()
        assert unions.eq(bboxes_a | bboxes_b).all()

        ious = intersections / unions
        assert bboxes_a.ious(bboxes_b).eq(ious).all()
        # yapf: enable


class TestBBoxesXY:

    @pytest.fixture
    def bboxes_xyxy(self) -> BBoxesXYXY:
        return BBoxesXYXY(torch.tensor([[10.0, 20.0, 40.0, 100.0]]))

    @pytest.fixture
    def bboxes_xywh(self) -> BBoxesXYWH:
        return BBoxesXYWH(torch.tensor([[10.0, 20.0, 30.0, 80.0]]))

    @pytest.mark.parametrize(
        'bboxes',
        ['bboxes_xyxy', 'bboxes_xywh'],
    )
    def test_properties(
        self,
        bboxes: str,
        request: pytest.FixtureRequest,
    ) -> None:
        bboxes_: BBoxesXY = request.getfixturevalue(bboxes)
        assert torch.tensor([10.0]).eq(bboxes_.left).all()
        assert torch.tensor([40.0]).eq(bboxes_.right).all()
        assert torch.tensor([20.0]).eq(bboxes_.top).all()
        assert torch.tensor([100.0]).eq(bboxes_.bottom).all()
        assert torch.tensor([30.0]).eq(bboxes_.width).all()
        assert torch.tensor([80.0]).eq(bboxes_.height).all()
        assert torch.tensor([25.0]).eq(bboxes_.center_x).all()
        assert torch.tensor([60.0]).eq(bboxes_.center_y).all()
        assert torch.tensor([10.0, 20.0]).eq(bboxes_.lt).all()
        assert torch.tensor([40.0, 100.0]).eq(bboxes_.rb).all()
        assert torch.tensor([30.0, 80.0]).eq(bboxes_.wh).all()
        assert torch.tensor([25.0, 60.0]).eq(bboxes_.center).all()
        assert torch.tensor([2400.0]).eq(bboxes_.area).all()


class TestBBoxesXYXY:

    def test_round(self):
        bboxes = BBoxesXYXY(torch.tensor([[10.9, 20.3, 39.2, 99.8]]))
        rounded = torch.tensor([10.0, 20.0, 40.0, 100.0])
        assert bboxes.round().to_tensor().eq(rounded).all()

    def test_expand(self):
        bboxes = BBoxesXYXY(torch.tensor([[40.0, 70.0, 60.0, 130.0]]))
        rounded = torch.tensor([35.0, 55.0, 65.0, 145.0])
        assert bboxes.expand(1.5).to_tensor().eq(rounded).all()

        bboxes = BBoxesXYXY(torch.tensor([[2.0, 10.0, 22.0, 70.0]]))
        rounded = torch.tensor([0.0, 0.0, 27.0, 85.0])
        assert bboxes.expand(1.5).to_tensor().eq(rounded).all()

        bboxes = BBoxesXYXY(torch.tensor([[40.0, 70.0, 60.0, 130.0]]))
        rounded = torch.tensor([35.0, 55.0, 65.0, 140.0])
        assert (
            bboxes.expand(1.5, image_wh=(70, 140))  # yapf: disable
            .to_tensor().eq(rounded).all()
        )


class TestBBoxesXYWH:

    def test_round(self):
        bboxes = BBoxesXYWH(torch.tensor([[10.9, 20.3, 28.3, 79.5]]))
        rounded = torch.tensor([10.0, 20.0, 30.0, 80.0])
        assert bboxes.round().to_tensor().eq(rounded).all()

    def test_expand(self):
        bboxes = BBoxesXYWH(torch.tensor([[40.0, 70.0, 20.0, 60.0]]))
        rounded = torch.tensor([35.0, 55.0, 30.0, 90.0])
        assert bboxes.expand(1.5).to_tensor().eq(rounded).all()

        bboxes = BBoxesXYWH(torch.tensor([[2.0, 10.0, 20.0, 60.0]]))
        rounded = torch.tensor([0.0, 0.0, 27.0, 85.0])
        assert bboxes.expand(1.5).to_tensor().eq(rounded).all()

        bboxes = BBoxesXYWH(torch.tensor([[40.0, 70.0, 20.0, 60.0]]))
        rounded = torch.tensor([35.0, 55.0, 30.0, 85.0])
        assert (
            bboxes.expand(1.5, image_wh=(70, 140))  # yapf: disable
            .to_tensor().eq(rounded).all()
        )
