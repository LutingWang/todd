import pytest
import torch

from todd.base.bboxes import BBoxes, BBoxesCXCYWH, BBoxesXYWH, BBoxesXYXY


class TestBBoxes:

    def test_ious(self) -> None:
        a = torch.tensor([
            [10.0, 20.0, 40.0, 100.0],
        ])
        b = torch.tensor([
            [5.0, 15.0, 8.0, 18.0],
            [5.0, 15.0, 8.0, 60.0],
            [5.0, 15.0, 8.0, 105.0],
            [5.0, 60.0, 8.0, 80.0],
            [5.0, 60.0, 8.0, 105.0],
            [5.0, 102.0, 8.0, 105.0],
            [5.0, 15.0, 25.0, 18.0],
            [5.0, 15.0, 25.0, 60.0],
            [5.0, 15.0, 25.0, 105.0],
            [5.0, 60.0, 25.0, 80.0],
            [5.0, 60.0, 25.0, 105.0],
            [5.0, 102.0, 25.0, 105.0],
            [5.0, 15.0, 45.0, 18.0],
            [5.0, 15.0, 45.0, 60.0],
            [5.0, 15.0, 45.0, 105.0],
            [5.0, 60.0, 45.0, 80.0],
            [5.0, 60.0, 45.0, 105.0],
            [5.0, 102.0, 45.0, 105.0],
            [25.0, 15.0, 35.0, 18.0],
            [25.0, 15.0, 35.0, 60.0],
            [25.0, 15.0, 35.0, 105.0],
            [25.0, 60.0, 35.0, 80.0],
            [25.0, 60.0, 35.0, 105.0],
            [25.0, 102.0, 35.0, 105.0],
            [25.0, 15.0, 45.0, 18.0],
            [25.0, 15.0, 45.0, 60.0],
            [25.0, 15.0, 45.0, 105.0],
            [25.0, 60.0, 45.0, 80.0],
            [25.0, 60.0, 45.0, 105.0],
            [25.0, 102.0, 45.0, 105.0],
            [42.0, 15.0, 45.0, 18.0],
            [42.0, 15.0, 45.0, 60.0],
            [42.0, 15.0, 45.0, 105.0],
            [42.0, 60.0, 45.0, 80.0],
            [42.0, 60.0, 45.0, 105.0],
            [42.0, 102.0, 45.0, 105.0],
        ])

        bboxes_a = BBoxesXYXY(a)
        bboxes_b = BBoxesXYXY(b)

        intersections = torch.tensor([[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 600.0, 1200.0, 300.0, 600.0,
            0.0, 0.0, 1200.0, 2400.0, 600.0, 1200.0, 0.0, 0.0, 400.0, 800.0,
            200.0, 400.0, 0.0, 0.0, 600.0, 1200.0, 300.0, 600.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ]])
        assert intersections.eq(bboxes_a & bboxes_b).all()

        unions = torch.tensor([[
            2409.0, 2535.0, 2670.0, 2460.0, 2535.0, 2409.0, 2460.0, 2700.0,
            3000.0, 2500.0, 2700.0, 2460.0, 2520.0, 3000.0, 3600.0, 2600.0,
            3000.0, 2520.0, 2430.0, 2450.0, 2500.0, 2400.0, 2450.0, 2430.0,
            2460.0, 2700.0, 3000.0, 2500.0, 2700.0, 2460.0, 2409.0, 2535.0,
            2670.0, 2460.0, 2535.0, 2409.0
        ]])
        assert bboxes_a.unions(bboxes_b, intersections).eq(unions).all()
        assert unions.eq(bboxes_a | bboxes_b).all()

        assert bboxes_a.ious(bboxes_b).eq(intersections / unions).all()

    @pytest.mark.parametrize(
        'bboxes',
        [
            BBoxesXYXY(torch.tensor([[10.0, 20.0, 40.0, 100.0]])),
            BBoxesXYWH(torch.tensor([[10.0, 20.0, 30.0, 80.0]])),
            BBoxesCXCYWH(torch.tensor([[25.0, 60.0, 30.0, 80.0]])),
        ],
    )
    def test_properties(self, bboxes: BBoxes) -> None:
        assert torch.tensor([10.0]).eq(bboxes.left).all()
        assert torch.tensor([40.0]).eq(bboxes.right).all()
        assert torch.tensor([20.0]).eq(bboxes.top).all()
        assert torch.tensor([100.0]).eq(bboxes.bottom).all()
        assert torch.tensor([30.0]).eq(bboxes.width).all()
        assert torch.tensor([80.0]).eq(bboxes.height).all()
        assert torch.tensor([25.0]).eq(bboxes.center_x).all()
        assert torch.tensor([60.0]).eq(bboxes.center_y).all()
        assert torch.tensor([10.0, 20.0]).eq(bboxes.lt).all()
        assert torch.tensor([40.0, 100.0]).eq(bboxes.rb).all()
        assert torch.tensor([30.0, 80.0]).eq(bboxes.wh).all()
        assert torch.tensor([25.0, 60.0]).eq(bboxes.center).all()
        assert torch.tensor([2400.0]).eq(bboxes.area).all()


class TestBBoxesXYXY:

    def test_round(self) -> None:
        bboxes = BBoxesXYXY(torch.tensor([[10.9, 20.3, 39.2, 99.8]]))
        rounded = torch.tensor([10.0, 20.0, 40.0, 100.0])
        assert bboxes.round().to_tensor().eq(rounded).all()

    def test_expand(self) -> None:
        bboxes = BBoxesXYXY(torch.tensor([[40.0, 70.0, 60.0, 130.0]]))
        expanded = torch.tensor([35.0, 55.0, 65.0, 145.0])
        assert bboxes.expand((1.5, 1.5)).to_tensor().eq(expanded).all()

        bboxes = BBoxesXYXY(torch.tensor([[2.0, 10.0, 22.0, 70.0]]))
        expanded = torch.tensor([-3.0, -5.0, 27.0, 85.0])
        assert bboxes.expand((1.5, 1.5)).to_tensor().eq(expanded).all()

    def test_clamp(self) -> None:
        bboxes = BBoxesXYXY(torch.tensor([[35.0, 55.0, 65.0, 145.0]]))
        clamped = torch.tensor([35.0, 55.0, 65.0, 140.0])
        assert bboxes.clamp((70, 140)).to_tensor().eq(clamped).all()

    def test_scale(self) -> None:
        bboxes = BBoxesXYXY(torch.tensor([[35.0, 55.0, 65.0, 145.0]]))
        scaled = torch.tensor([70.0, 110.0, 130.0, 290.0])
        assert bboxes.scale((2, 2)).to_tensor().eq(scaled).all()

        bboxes = BBoxesXYXY(torch.tensor([[2.0, 10.0, 22.0, 70.0]]))
        scaled = torch.tensor([1.0, 20.0, 11.0, 140.0])
        assert bboxes.scale((0.5, 2.0)).to_tensor().eq(scaled).all()


class TestBBoxesXYWH:

    def test_round(self):
        bboxes = BBoxesXYWH(torch.tensor([[10.9, 20.3, 28.3, 79.5]]))
        rounded = torch.tensor([10.0, 20.0, 30.0, 80.0])
        assert bboxes.round().to_tensor().eq(rounded).all()

    def test_expand(self):
        bboxes = BBoxesXYWH(torch.tensor([[40.0, 70.0, 20.0, 60.0]]))
        rounded = torch.tensor([35.0, 55.0, 30.0, 90.0])
        assert bboxes.expand((1.5, 1.5)).to_tensor().eq(rounded).all()

        bboxes = BBoxesXYWH(torch.tensor([[2.0, 10.0, 20.0, 60.0]]))
        rounded = torch.tensor([-3.0, -5.0, 30.0, 90.0])
        assert bboxes.expand((1.5, 1.5)).to_tensor().eq(rounded).all()

    def test_clamp(self) -> None:
        bboxes = BBoxesXYWH(torch.tensor([[35.0, 55.0, 30.0, 90.0]]))
        rounded = torch.tensor([35.0, 55.0, 30.0, 85.0])
        assert bboxes.clamp((70, 140)).to_tensor().eq(rounded).all()

    def test_scale(self) -> None:
        bboxes = BBoxesXYWH(torch.tensor([[35.0, 55.0, 30.0, 90.0]]))
        scaled = torch.tensor([70.0, 110.0, 60.0, 180.0])
        assert bboxes.scale((2, 2)).to_tensor().eq(scaled).all()

        bboxes = BBoxesXYWH(torch.tensor([[2.0, 10.0, 20.0, 60.0]]))
        scaled = torch.tensor([1.0, 20.0, 10.0, 120.0])
        assert bboxes.scale((0.5, 2.0)).to_tensor().eq(scaled).all()
