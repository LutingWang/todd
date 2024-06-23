# pylint: disable=invalid-name

import einops
import pytest
import torch

from todd.tasks import FlattenMixin
from todd.tasks.object_detection.bboxes import (
    BBox,
    BBoxes,
    BBoxesCXCYWH,
    BBoxesXYWH,
    BBoxesXYXY,
    FlattenBBoxesXYXY,
)

tensor1 = torch.tensor([10., 20., 40., 100.])
tensor2 = torch.tensor([
    [5., 15., 8., 18.],
    [5., 15., 8., 60.],
    [5., 15., 8., 105.],
    [5., 60., 8., 80.],
    [5., 60., 8., 105.],
    [5., 102., 8., 105.],
    [5., 15., 25., 18.],
    [5., 15., 25., 60.],
    [5., 15., 25., 105.],
    [5., 60., 25., 80.],
    [5., 60., 25., 105.],
    [5., 102., 25., 105.],
    [5., 15., 45., 18.],
    [5., 15., 45., 60.],
    [5., 15., 45., 105.],
    [5., 60., 45., 80.],
    [5., 60., 45., 105.],
    [5., 102., 45., 105.],
    [25., 15., 35., 18.],
    [25., 15., 35., 60.],
    [25., 15., 35., 105.],
    [25., 60., 35., 80.],
    [25., 60., 35., 105.],
    [25., 102., 35., 105.],
    [25., 15., 45., 18.],
    [25., 15., 45., 60.],
    [25., 15., 45., 105.],
    [25., 60., 45., 80.],
    [25., 60., 45., 105.],
    [25., 102., 45., 105.],
    [42., 15., 45., 18.],
    [42., 15., 45., 60.],
    [42., 15., 45., 105.],
    [42., 60., 45., 80.],
    [42., 60., 45., 105.],
    [42., 102., 45., 105.],
])
intersections = torch.tensor([
    0., 0., 0., 0., 0., 0., 0., 600., 1200., 300., 600., 0., 0., 1200., 2400.,
    600., 1200., 0., 0., 400., 800., 200., 400., 0., 0., 600., 1200., 300.,
    600., 0., 0., 0., 0., 0., 0., 0.
])
unions = torch.tensor([
    2409., 2535., 2670., 2460., 2535., 2409., 2460., 2700., 3000., 2500.,
    2700., 2460., 2520., 3000., 3600., 2600., 3000., 2520., 2430., 2450.,
    2500., 2400., 2450., 2430., 2460., 2700., 3000., 2500., 2700., 2460.,
    2409., 2535., 2670., 2460., 2535., 2409.
])


class ConcreteBBoxes(BBoxes):

    def flatten(self) -> FlattenMixin[BBox]:
        raise NotImplementedError

    @property
    def left(self) -> torch.Tensor:
        return torch.tensor([10.])

    @property
    def right(self) -> torch.Tensor:
        return torch.tensor([30.])

    @property
    def top(self) -> torch.Tensor:
        return torch.tensor([20.])

    @property
    def bottom(self) -> torch.Tensor:
        return torch.tensor([50.])

    @property
    def width(self) -> torch.Tensor:
        return torch.tensor([20.])

    @property
    def height(self) -> torch.Tensor:
        return torch.tensor([30.])

    @property
    def center_x(self) -> torch.Tensor:
        return torch.tensor([20.])

    @property
    def center_y(self) -> torch.Tensor:
        return torch.tensor([35.])

    @property
    def lt(self) -> torch.Tensor:
        return torch.tensor([[10., 20.]])

    @property
    def rb(self) -> torch.Tensor:
        return torch.tensor([[30., 50.]])

    @property
    def wh(self) -> torch.Tensor:
        return torch.tensor([[20., 30.]])

    @property
    def center(self) -> torch.Tensor:
        return torch.tensor([[20., 35.]])

    @classmethod
    def _from1(cls, bboxes: BBoxes) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def _from2(cls, bboxes: BBoxes) -> torch.Tensor:
        raise NotImplementedError


class ConcreteBBoxes1(ConcreteBBoxes):

    @classmethod
    def _from1(cls, bboxes: BBoxes) -> torch.Tensor:
        return torch.tensor([[100., 200.]])

    @classmethod
    def _from2(cls, bboxes: BBoxes) -> torch.Tensor:
        return torch.tensor([[300., 500.]])


class TestBBoxes:

    @pytest.fixture(scope='class')
    def bboxes1(self) -> BBoxesXYXY:
        return BBoxesXYXY(torch.tensor([[10., 20., 30., 50.]]))

    @pytest.fixture(scope='class')
    def bboxes2(self) -> BBoxesXYXY:
        return BBoxesXYXY(
            torch.tensor([[10., 20., 30., 50.], [100., 200., 300., 500.]])
        )

    @pytest.fixture(scope='class')
    def bboxes3(self) -> BBoxesXYXY:
        return BBoxesXYXY(torch.tensor([[10.9, 20.3, 30.1, 50.5]]))

    @pytest.fixture(scope='class')
    def bboxes4(self) -> BBoxesXYXY:
        return BBoxesXYXY(
            torch.tensor([
                [-30., -50., -10., -20.],
                [-10., -20., 30., 50.],
                [-10., -20., 300., 500.],
                [10., 20., 30., 50.],
                [10., 20., 300., 500.],
                [110., 210., 300., 500.],
            ]),
        )

    @pytest.fixture(scope='class')
    def bboxes5(self) -> BBoxesXYXY:
        tensor = einops.repeat(tensor1, 'd -> n d', n=36)
        return BBoxesXYXY(tensor)

    @pytest.fixture(scope='class')
    def bboxes6(self) -> BBoxesXYXY:
        return BBoxesXYXY(tensor2)

    def test_to_object(self) -> None:
        tensor = torch.tensor([10., 20., 30., 50.])
        assert BBoxes.to_object(tensor) == (10., 20., 30., 50.)

    def test_init(self) -> None:
        with pytest.raises(AssertionError):
            ConcreteBBoxes(torch.tensor([[10., 20., 30.]]))

    def test_scale(self, bboxes1: BBoxesXYXY) -> None:
        bboxes1 = bboxes1.scale((2., 4.))
        target = torch.tensor([[20., 80., 60., 200.]])
        assert torch.allclose(bboxes1.to_tensor(), target)

    def test_area(self, bboxes1: BBoxesXYXY) -> None:
        assert torch.allclose(bboxes1.area, torch.tensor(600.))

    def test_from(self, bboxes1: BBoxesXYXY) -> None:
        bboxes1_ = ConcreteBBoxes1.from_(bboxes1)
        target = torch.tensor([[100., 200., 300., 500.]])
        assert isinstance(bboxes1_, ConcreteBBoxes1)
        assert torch.allclose(bboxes1_.to_tensor(), target)

    def test_to(self, bboxes1: BBoxesXYXY) -> None:
        bboxes1_ = bboxes1.to(ConcreteBBoxes1)
        target = torch.tensor([[100., 200., 300., 500.]])
        assert isinstance(bboxes1_, ConcreteBBoxes1)
        assert torch.allclose(bboxes1_.to_tensor(), target)

    def test_translate_tuple(self, bboxes2: BBoxesXYXY) -> None:
        bboxes2 = bboxes2.translate((10., 20.))
        target = torch.tensor([[20., 40., 40., 70.], [110., 220., 310., 520.]])
        assert isinstance(bboxes2, BBoxesXYXY)
        assert torch.allclose(bboxes2.to_tensor(), target)

    def test_translate_tensor1(self, bboxes2: BBoxesXYXY) -> None:
        bboxes2 = bboxes2.translate(torch.tensor([10., 20.]))
        target = torch.tensor([
            [20., 40., 40., 70.],
            [110., 220., 310., 520.],
        ])
        assert isinstance(bboxes2, BBoxesXYXY)
        assert torch.allclose(bboxes2.to_tensor(), target)

    def test_translate_tensor2(self, bboxes2: BBoxesXYXY) -> None:
        bboxes2 = bboxes2.translate(torch.tensor([[10., 20.], [100., 200.]]))
        target = torch.tensor([
            [20., 40., 40., 70.],
            [200., 400., 400., 700.],
        ])
        assert isinstance(bboxes2, BBoxesXYXY)
        assert torch.allclose(bboxes2.to_tensor(), target)

    def test_round(self, bboxes3: BBoxesXYXY) -> None:
        bboxes3 = bboxes3.round()
        target = torch.tensor([[10., 20., 31., 51.]])
        assert isinstance(bboxes3, BBoxesXYXY)
        assert torch.allclose(bboxes3.to_tensor(), target)

    def test_round_normalized(self, bboxes3: BBoxesXYXY) -> None:
        bboxes3.set_divisor((100., 200.))
        bboxes3 = bboxes3.normalize()
        bboxes3 = bboxes3.round()
        target = torch.tensor([[0.1, 0.1, 0.31, 0.255]])
        assert isinstance(bboxes3, BBoxesXYXY)
        assert bboxes3.normalized
        assert torch.allclose(bboxes3.to_tensor(), target)

    def test_expand_tuple(self, bboxes2: BBoxesXYXY) -> None:
        bboxes2 = bboxes2.expand((2., 1.5))
        target = torch.tensor([
            [0., 35. - 45. / 2., 40., 35. + 45. / 2.],
            [0., 350. - 450. / 2., 400., 350. + 450. / 2.],
        ])
        assert isinstance(bboxes2, BBoxesXYXY)
        assert torch.allclose(bboxes2.to_tensor(), target)

    def test_expand_tensor1(self, bboxes2: BBoxesXYXY) -> None:
        bboxes2 = bboxes2.expand(torch.tensor([2., 1.5]))
        target = torch.tensor([
            [0., 35. - 45. / 2., 40., 35. + 45. / 2.],
            [0., 350. - 450. / 2., 400., 350. + 450. / 2.],
        ])
        assert isinstance(bboxes2, BBoxesXYXY)
        assert torch.allclose(bboxes2.to_tensor(), target)

    def test_expand_tensor2(self, bboxes2: BBoxesXYXY) -> None:
        bboxes2 = bboxes2.expand(torch.tensor([[2., 1.5], [3., 2.]]))
        target = torch.tensor([
            [0., 35. - 45. / 2., 40., 35. + 45. / 2.],
            [-100., 50., 500., 650.],
        ])
        assert isinstance(bboxes2, BBoxesXYXY)
        assert torch.allclose(bboxes2.to_tensor(), target)

    def test_clamp1(self, bboxes4: BBoxesXYXY) -> None:
        with pytest.raises(AttributeError):
            bboxes4.clamp()

    def test_clamp2(self, bboxes4: BBoxesXYXY) -> None:
        bboxes4.set_divisor((100., 200.))
        bboxes4 = bboxes4.clamp()
        tensor = torch.tensor([
            [0., 0., 0., 0.],
            [0., 0., 30., 50.],
            [0., 0., 100., 200.],
            [10., 20., 30., 50.],
            [10., 20., 100., 200.],
            [100., 200., 100., 200.],
        ])
        assert isinstance(bboxes4, BBoxesXYXY)
        assert torch.allclose(bboxes4.to_tensor(), tensor)

    def test_clamp_normalized(self, bboxes4: BBoxesXYXY) -> None:
        bboxes4.set_divisor((100., 200.))
        bboxes4 = bboxes4.normalize()
        bboxes4 = bboxes4.clamp()
        tensor = torch.tensor([
            [0., 0., 0., 0.],
            [0., 0., 0.3, 0.25],
            [0., 0., 1., 1.],
            [0.1, 0.1, 0.3, 0.25],
            [0.1, 0.1, 1., 1.],
            [1., 1., 1., 1.],
        ])
        assert isinstance(bboxes4, BBoxesXYXY)
        assert bboxes4.normalized
        assert torch.allclose(bboxes4.to_tensor(), tensor)

    def test_indices(self, bboxes2: BBoxesXYXY) -> None:
        target = torch.tensor([True, True], dtype=torch.bool)
        assert torch.allclose(bboxes2.indices(), target)

        target = torch.tensor([False, True], dtype=torch.bool)
        assert torch.allclose(bboxes2.indices(min_area=1000.), target)

        target = torch.tensor([False, True], dtype=torch.bool)
        assert torch.allclose(bboxes2.indices(min_wh=(100., 100.)), target)

        target = torch.tensor([False, False], dtype=torch.bool)
        assert (
            bboxes2.indices(
                min_area=1000.,
                min_wh=(1000., 1000.),
            ) == target
        ).all()

    def test_pairwise_intersections(
        self,
        bboxes5: BBoxesXYXY,
        bboxes6: BBoxesXYXY,
    ) -> None:
        assert torch.allclose(
            bboxes5.pairwise_intersections(bboxes6), intersections
        )

    def test__pairwise_unions(
        self,
        bboxes5: BBoxesXYXY,
        bboxes6: BBoxesXYXY,
    ) -> None:
        assert (bboxes5._pairwise_unions(
            bboxes6,
            intersections,
        ) == unions).all()

    def test_pairwise_unions(
        self,
        bboxes5: BBoxesXYXY,
        bboxes6: BBoxesXYXY,
    ) -> None:
        assert torch.allclose(bboxes5.pairwise_unions(bboxes6), unions)

    def test_pairwise_ious(
        self,
        bboxes5: BBoxesXYXY,
        bboxes6: BBoxesXYXY,
    ) -> None:
        pairwise_ious = intersections / unions
        assert torch.allclose(bboxes5.pairwise_ious(bboxes6), pairwise_ious)


class TestBBoxesXY__:

    @pytest.fixture(scope='class')
    def bboxes1(self) -> BBoxesXYXY:
        return BBoxesXYXY(torch.tensor([[10., 20., 30., 50.]]))

    def test_left(self, bboxes1: BBoxesXYXY) -> None:
        assert bboxes1.left.item() == 10.

    def test_top(self, bboxes1: BBoxesXYXY) -> None:
        assert bboxes1.top.item() == 20.

    def test_lt(self, bboxes1: BBoxesXYXY) -> None:
        assert bboxes1.lt.tolist() == [[10., 20.]]

    def test_from1(self, bboxes1: BBoxesXYXY) -> None:
        assert BBoxesXYXY._from1(bboxes1).tolist() == [[10., 20.]]


class TestBBoxesCXCY__:

    @pytest.fixture(scope='class')
    def bboxes1(self) -> BBoxesCXCYWH:
        return BBoxesCXCYWH(torch.tensor([[10., 20., 30., 50.]]))

    def test_center_x(self, bboxes1: BBoxesCXCYWH) -> None:
        assert bboxes1.center_x.item() == 10.

    def test_center_y(self, bboxes1: BBoxesCXCYWH) -> None:
        assert bboxes1.center_y.item() == 20.

    def test_center(self, bboxes1: BBoxesCXCYWH) -> None:
        assert bboxes1.center.tolist() == [[10., 20.]]

    def test_from1(self, bboxes1: BBoxesCXCYWH) -> None:
        assert BBoxesCXCYWH._from1(bboxes1).tolist() == [[10., 20.]]


class TestBBoxes__XY:  # noqa: N801

    @pytest.fixture(scope='class')
    def bboxes1(self) -> BBoxesXYXY:
        return BBoxesXYXY(torch.tensor([[10., 20., 30., 50.]]))

    def test_right(self, bboxes1: BBoxesXYXY) -> None:
        assert bboxes1.right.item() == 30.

    def test_bottom(self, bboxes1: BBoxesXYXY) -> None:
        assert bboxes1.bottom.item() == 50.

    def test_rb(self, bboxes1: BBoxesXYXY) -> None:
        assert bboxes1.rb.tolist() == [[30., 50.]]

    def test_from2(self, bboxes1: BBoxesXYXY) -> None:
        assert BBoxesXYXY._from2(bboxes1).tolist() == [[30., 50.]]


class TestBBoxes__WH:  # noqa: N801

    @pytest.fixture(scope='class')
    def bboxes1(self) -> BBoxesXYWH:
        return BBoxesXYWH(torch.tensor([[10., 20., 30., 50.]]))

    def test_width(self, bboxes1: BBoxesXYWH) -> None:
        assert bboxes1.width.item() == 30.

    def test_height(self, bboxes1: BBoxesXYWH) -> None:
        assert bboxes1.height.item() == 50.

    def test_wh(self, bboxes1: BBoxesXYWH) -> None:
        assert bboxes1.wh.tolist() == [[30., 50.]]

    def test_from2(self, bboxes1: BBoxesXYWH) -> None:
        assert BBoxesXYWH._from2(bboxes1).tolist() == [[30., 50.]]


class TestBBoxesXYXY:

    @pytest.fixture(scope='class')
    def bboxes1(self) -> BBoxesXYXY:
        return BBoxesXYXY(torch.tensor([[10., 20., 30., 50.]]))

    def test_width(self, bboxes1: BBoxesXYXY) -> None:
        assert bboxes1.width.item() == 20.

    def test_height(self, bboxes1: BBoxesXYXY) -> None:
        assert bboxes1.height.item() == 30.

    def test_center_x(self, bboxes1: BBoxesXYXY) -> None:
        assert bboxes1.center_x.item() == 20.

    def test_center_y(self, bboxes1: BBoxesXYXY) -> None:
        assert bboxes1.center_y.item() == 35.

    def test_wh(self, bboxes1: BBoxesXYXY) -> None:
        assert bboxes1.wh.tolist() == [[20., 30.]]

    def test_center(self, bboxes1: BBoxesXYXY) -> None:
        assert bboxes1.center.tolist() == [[20., 35.]]


class TestBBoxesXYWH:

    @pytest.fixture(scope='class')
    def bboxes_(self) -> BBoxesXYWH:
        return BBoxesXYWH(torch.tensor([[10., 20., 20., 30.]]))

    def test_right(self, bboxes_: BBoxesXYWH) -> None:
        assert bboxes_.right.item() == 30.

    def test_bottom(self, bboxes_: BBoxesXYWH) -> None:
        assert bboxes_.bottom.item() == 50.

    def test_center_x(self, bboxes_: BBoxesXYWH) -> None:
        assert bboxes_.center_x.item() == 20.

    def test_center_y(self, bboxes_: BBoxesXYWH) -> None:
        assert bboxes_.center_y.item() == 35.

    def test_rb(self, bboxes_: BBoxesXYWH) -> None:
        assert bboxes_.rb.tolist() == [[30., 50.]]

    def test_center(self, bboxes_: BBoxesXYWH) -> None:
        assert bboxes_.center.tolist() == [[20., 35.]]


class TestBBBoxesCXCYWH:

    @pytest.fixture(scope='class')
    def bboxes_(self) -> BBoxesCXCYWH:
        return BBoxesCXCYWH(torch.tensor([[20., 35., 20., 30.]]))

    def test_left(self, bboxes_: BBoxesCXCYWH) -> None:
        assert bboxes_.left.item() == 10.

    def test_right(self, bboxes_: BBoxesCXCYWH) -> None:
        assert bboxes_.right.item() == 30.

    def test_top(self, bboxes_: BBoxesCXCYWH) -> None:
        assert bboxes_.top.item() == 20.

    def test_bottom(self, bboxes_: BBoxesCXCYWH) -> None:
        assert bboxes_.bottom.item() == 50.

    def test_lt(self, bboxes_: BBoxesCXCYWH) -> None:
        assert bboxes_.lt.tolist() == [[10., 20.]]

    def test_rb(self, bboxes_: BBoxesCXCYWH) -> None:
        assert bboxes_.rb.tolist() == [[30., 50.]]


class TestFlattenMixin:

    @pytest.fixture(scope='class')
    def bboxes1(self) -> FlattenBBoxesXYXY:
        tensor = einops.rearrange(tensor1, 'd -> 1 d')
        return FlattenBBoxesXYXY(tensor)

    @pytest.fixture(scope='class')
    def bboxes2(self) -> FlattenBBoxesXYXY:
        return FlattenBBoxesXYXY(tensor2)

    def test_intersections(
        self,
        bboxes1: FlattenBBoxesXYXY,
        bboxes2: FlattenBBoxesXYXY,
    ) -> None:
        target = einops.rearrange(intersections, 'd -> 1 d')
        assert torch.allclose(bboxes1.pairwise_intersections(bboxes2), target)

    def test__unions(
        self,
        bboxes1: FlattenBBoxesXYXY,
        bboxes2: FlattenBBoxesXYXY,
    ) -> None:
        target = einops.rearrange(unions, 'd -> 1 d')
        assert (
            bboxes1._unions(
                bboxes2,
                einops.rearrange(intersections, 'd -> 1 d'),
            ) == target
        ).all()

    def test_unions(
        self,
        bboxes1: FlattenBBoxesXYXY,
        bboxes2: FlattenBBoxesXYXY,
    ) -> None:
        target = einops.rearrange(unions, 'd -> 1 d')
        assert torch.allclose(bboxes1.unions(bboxes2), target)

    def test_ious(
        self,
        bboxes1: FlattenBBoxesXYXY,
        bboxes2: FlattenBBoxesXYXY,
    ) -> None:
        ious = intersections / unions
        target = einops.rearrange(ious, 'd -> 1 d')
        assert torch.allclose(bboxes1.ious(bboxes2), target)
