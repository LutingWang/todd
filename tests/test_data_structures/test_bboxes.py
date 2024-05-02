# pylint: disable=invalid-name,pointless-statement

import pytest
import torch

from todd.data_structures.bboxes import (
    BBoxes,
    BBoxes__WH,
    BBoxes__XY,
    BBoxesCXCY__,
    BBoxesCXCYWH,
    BBoxesXY__,
    BBoxesXYWH,
    BBoxesXYXY,
)


class PseudoBBoxes(BBoxes):

    @property
    def left(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def right(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def top(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def bottom(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def width(self) -> torch.Tensor:
        return torch.tensor([[50.0]])

    @property
    def height(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def center_x(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def center_y(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def lt(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def rb(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def wh(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def center(self) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def _from1(cls, bboxes: BBoxes) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def _from2(cls, bboxes: BBoxes) -> torch.Tensor:
        raise NotImplementedError


class TestBBoxes:

    def test_init(self) -> None:
        with pytest.raises(ValueError):
            PseudoBBoxes(torch.tensor([10.0, 20.0, 30.0, 50.0]))
        with pytest.raises(ValueError):
            PseudoBBoxes(torch.tensor([[[10.0, 20.0, 30.0, 50.0]]]))
        with pytest.raises(ValueError):
            PseudoBBoxes(torch.tensor([[10.0, 20.0]]))

    def test_len(self) -> None:
        assert len(PseudoBBoxes(torch.rand(0, 4))) == 0
        assert len(PseudoBBoxes(torch.rand(1, 4))) == 1
        assert len(PseudoBBoxes(torch.rand(2, 4))) == 2

    def test_iter(self) -> None:
        bboxes = torch.rand(10, 4)
        for bbox1, bbox2 in zip(bboxes, PseudoBBoxes(bboxes)):
            assert bbox1.tolist() == list(bbox2)

    def test_copy(self) -> None:
        bboxes1 = torch.rand(13, 4)
        bboxes2 = torch.rand(17, 4)

        bboxes1_ = PseudoBBoxes(bboxes1)
        bboxes2_ = bboxes1_._copy(bboxes2)
        assert bboxes2_.to_tensor().shape == (17, 4)
        assert not bboxes2_.normalized
        assert not bboxes2_.has_image_wh

        bboxes1_ = PseudoBBoxes(bboxes1, normalized=True)
        bboxes2_ = bboxes1_._copy(bboxes2)
        assert bboxes2_.to_tensor().shape == (17, 4)
        assert bboxes2_.normalized
        assert not bboxes2_.has_image_wh

        bboxes1_ = PseudoBBoxes(bboxes1, image_wh=(10, 10))
        bboxes2_ = bboxes1_._copy(bboxes2)
        assert bboxes2_.to_tensor().shape == (17, 4)
        assert not bboxes2_.normalized
        assert bboxes2_.image_wh == (10, 10)

        bboxes1_ = PseudoBBoxes(bboxes1, normalized=True, image_wh=(10, 10))
        bboxes2_ = bboxes1_._copy(bboxes2)
        assert bboxes2_.to_tensor().shape == (17, 4)
        assert bboxes2_.normalized
        assert bboxes2_.image_wh == (10, 10)

    def test_getitem(self) -> None:
        bboxes = torch.rand(10, 4)
        bboxes_ = PseudoBBoxes(bboxes)
        assert bboxes_[0].to_tensor().eq(bboxes[[0]]).all()
        assert bboxes_[[0]].to_tensor().eq(bboxes[[0]]).all()
        assert bboxes_[[0, 1]].to_tensor().eq(bboxes[[0, 1]]).all()
        assert bboxes_[:2].to_tensor().eq(bboxes[:2]).all()
        assert bboxes_[2:].to_tensor().eq(bboxes[2:]).all()

        with pytest.raises(ValueError):
            bboxes_[0, 1]
        with pytest.raises(ValueError):
            bboxes_[None]

    def test_add(self) -> None:
        bboxes1 = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes2 = torch.tensor([[30.0, 50.0, 50.0, 70.0]])
        bboxes = torch.cat((bboxes1, bboxes2))

        bboxes1_ = PseudoBBoxes(bboxes1)
        bboxes2_ = PseudoBBoxes(bboxes2)
        added_bboxes = bboxes1_ + bboxes2_
        assert added_bboxes.to_tensor().eq(bboxes).all()
        assert not added_bboxes.normalized
        assert not added_bboxes.has_image_wh

        bboxes1_ = PseudoBBoxes(bboxes1, normalized=True)
        bboxes2_ = PseudoBBoxes(bboxes2, normalized=True)
        added_bboxes = bboxes1_ + bboxes2_
        assert added_bboxes.to_tensor().eq(bboxes).all()
        assert added_bboxes.normalized
        assert not added_bboxes.has_image_wh

        bboxes1_ = PseudoBBoxes(bboxes1, image_wh=(10, 10))
        bboxes2_ = PseudoBBoxes(bboxes2, image_wh=(10, 10))
        added_bboxes = bboxes1_ + bboxes2_
        assert added_bboxes.to_tensor().eq(bboxes).all()
        assert not added_bboxes.normalized
        assert added_bboxes.image_wh == (10, 10)

        bboxes1_ = PseudoBBoxes(bboxes1, normalized=True, image_wh=(10, 10))
        bboxes2_ = PseudoBBoxes(bboxes2, normalized=True, image_wh=(10, 10))
        added_bboxes = bboxes1_ + bboxes2_
        assert added_bboxes.to_tensor().eq(bboxes).all()
        assert added_bboxes.normalized
        assert added_bboxes.image_wh == (10, 10)

        bboxes1_ = PseudoBBoxes(bboxes1)
        bboxes2_ = PseudoBBoxes(bboxes2, normalized=True)
        with pytest.raises(AssertionError):
            bboxes1_ + bboxes2_

        bboxes1_ = PseudoBBoxes(bboxes1, normalized=True)
        bboxes2_ = PseudoBBoxes(bboxes2)
        with pytest.raises(AssertionError):
            bboxes1_ + bboxes2_

        bboxes1_ = PseudoBBoxes(bboxes1)
        bboxes2_ = PseudoBBoxes(bboxes2, image_wh=(10, 10))
        with pytest.raises(AssertionError):
            bboxes1_ + bboxes2_

        bboxes1_ = PseudoBBoxes(bboxes1, image_wh=(10, 10))
        bboxes2_ = PseudoBBoxes(bboxes2)
        with pytest.raises(AttributeError):
            bboxes1_ + bboxes2_

        bboxes1_ = PseudoBBoxes(bboxes1, image_wh=(10, 10))
        bboxes2_ = PseudoBBoxes(bboxes2, image_wh=(10, 20))
        with pytest.raises(AssertionError):
            bboxes1_ + bboxes2_

    def test_normalized(self) -> None:
        bboxes = torch.rand(10, 4)
        bboxes_ = PseudoBBoxes(bboxes)
        assert not bboxes_.normalized
        bboxes_ = PseudoBBoxes(bboxes, normalized=True)
        assert bboxes_.normalized

    def test_has_image_wh(self) -> None:
        bboxes = torch.rand(10, 4)
        bboxes_ = PseudoBBoxes(bboxes)
        assert not bboxes_.has_image_wh
        bboxes_ = PseudoBBoxes(bboxes, image_wh=(10, 10))
        assert bboxes_.has_image_wh

    def test_image_wh(self) -> None:
        bboxes = torch.rand(10, 4)
        bboxes_ = PseudoBBoxes(bboxes)
        with pytest.raises(AttributeError):
            bboxes_.image_wh
        bboxes_ = PseudoBBoxes(bboxes, image_wh=(10, 10))
        assert bboxes_.image_wh == (10, 10)

    def test_set_image_wh(self) -> None:
        bboxes = torch.rand(10, 4)
        bboxes_ = PseudoBBoxes(bboxes)
        bboxes_.set_image_wh((10, 10))
        assert bboxes_.image_wh == (10, 10)
        bboxes_ = PseudoBBoxes(bboxes, image_wh=(5, 5))
        with pytest.raises(AssertionError):
            bboxes_.set_image_wh((10, 10))
        bboxes_ = PseudoBBoxes(bboxes, image_wh=(5, 5))
        bboxes_.set_image_wh((10, 10), True)
        assert bboxes_.image_wh == (10, 10)

    def test_to_tensor(self) -> None:
        bboxes = torch.rand(10, 4)
        bboxes_ = PseudoBBoxes(bboxes)
        assert bboxes_.to_tensor().eq(bboxes).all()

    def test_scale(self) -> None:
        bboxes = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_ = PseudoBBoxes(bboxes)
        scaler = torch.tensor([[2, 4, 2, 4]])
        assert bboxes_._scale((2, 4)).eq(scaler).all()
        scaled_bboxes = torch.tensor([[20, 80, 60, 200]])
        scaled_bboxes_ = bboxes_.scale((2, 4))
        assert scaled_bboxes_.to_tensor().eq(scaled_bboxes).all()

        bboxes_ = PseudoBBoxes(bboxes)
        scaled_bboxes_ = bboxes_.scale((2, 4))
        assert not scaled_bboxes_.normalized
        assert not scaled_bboxes_.has_image_wh

        bboxes_ = PseudoBBoxes(bboxes, normalized=True)
        scaled_bboxes_ = bboxes_.scale((2, 4))
        assert scaled_bboxes_.normalized
        assert not scaled_bboxes_.has_image_wh

        bboxes_ = PseudoBBoxes(bboxes, image_wh=(10, 10))
        scaled_bboxes_ = bboxes_.scale((2, 4))
        assert not scaled_bboxes_.normalized
        assert scaled_bboxes_.image_wh == (10, 10)

        bboxes_ = PseudoBBoxes(bboxes, normalized=True, image_wh=(10, 10))
        scaled_bboxes_ = bboxes_.scale((2, 4))
        assert scaled_bboxes_.normalized
        assert scaled_bboxes_.image_wh == (10, 10)

    def test_mul(self) -> None:
        bboxes = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_ = PseudoBBoxes(bboxes)
        mul_bboxes = torch.tensor([[20, 80, 60, 200]])
        mul_bboxes_ = bboxes_ * (2, 4)
        assert mul_bboxes_.to_tensor().eq(mul_bboxes).all()

        bboxes_ = PseudoBBoxes(bboxes)
        mul_bboxes_ = bboxes_ * (2, 4)
        assert not mul_bboxes_.normalized
        assert not mul_bboxes_.has_image_wh

        bboxes_ = PseudoBBoxes(bboxes, normalized=True)
        mul_bboxes_ = bboxes_ * (2, 4)
        assert mul_bboxes_.normalized
        assert not mul_bboxes_.has_image_wh

        bboxes_ = PseudoBBoxes(bboxes, image_wh=(10, 10))
        mul_bboxes_ = bboxes_ * (2, 4)
        assert not mul_bboxes_.normalized
        assert mul_bboxes_.image_wh == (10, 10)

        bboxes_ = PseudoBBoxes(bboxes, normalized=True, image_wh=(10, 10))
        mul_bboxes_ = bboxes_ * (2, 4)
        assert mul_bboxes_.normalized
        assert mul_bboxes_.image_wh == (10, 10)

    def test_truediv(self) -> None:
        bboxes = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_ = PseudoBBoxes(bboxes)
        mul_bboxes = torch.tensor([[5, 5, 15, 12.5]])
        mul_bboxes_ = bboxes_ / (2, 4)
        assert mul_bboxes_.to_tensor().eq(mul_bboxes).all()

        bboxes_ = PseudoBBoxes(bboxes)
        mul_bboxes_ = bboxes_ / (2, 4)
        assert not mul_bboxes_.normalized
        assert not mul_bboxes_.has_image_wh

        bboxes_ = PseudoBBoxes(bboxes, normalized=True)
        mul_bboxes_ = bboxes_ / (2, 4)
        assert mul_bboxes_.normalized
        assert not mul_bboxes_.has_image_wh

        bboxes_ = PseudoBBoxes(bboxes, image_wh=(10, 10))
        mul_bboxes_ = bboxes_ / (2, 4)
        assert not mul_bboxes_.normalized
        assert mul_bboxes_.image_wh == (10, 10)

        bboxes_ = PseudoBBoxes(bboxes, normalized=True, image_wh=(10, 10))
        mul_bboxes_ = bboxes_ * (2, 4)
        assert mul_bboxes_.normalized
        assert mul_bboxes_.image_wh == (10, 10)

    def test_translate(self) -> None:
        bboxes = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_ = PseudoBBoxes(bboxes)
        translation = torch.tensor([[10, 20, 10, 20]])
        assert bboxes_._translate((10, 20)).eq(translation).all()
        translated_bboxes = torch.tensor([[20, 40, 40, 70]])
        translated_bboxes_ = bboxes_.translate((10, 20))
        assert translated_bboxes_.to_tensor().eq(translated_bboxes).all()

        bboxes_ = PseudoBBoxes(bboxes)
        translated_bboxes_ = bboxes_.translate((10, 20))
        assert not translated_bboxes_.normalized
        assert not translated_bboxes_.has_image_wh

        bboxes_ = PseudoBBoxes(bboxes, normalized=True)
        translated_bboxes_ = bboxes_.translate((10, 20))
        assert translated_bboxes_.normalized
        assert not translated_bboxes_.has_image_wh

        bboxes_ = PseudoBBoxes(bboxes, image_wh=(10, 10))
        translated_bboxes_ = bboxes_.translate((10, 20))
        assert not translated_bboxes_.normalized
        assert translated_bboxes_.image_wh == (10, 10)

        bboxes_ = PseudoBBoxes(bboxes, normalized=True, image_wh=(10, 10))
        translated_bboxes_ = bboxes_.translate((10, 20))
        assert translated_bboxes_.normalized
        assert translated_bboxes_.image_wh == (10, 10)

    def test_normalize(self) -> None:
        bboxes = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_ = PseudoBBoxes(bboxes)
        with pytest.raises(AttributeError):
            bboxes_.normalize()

        bboxes_ = PseudoBBoxes(bboxes, normalized=True)
        normalized_bboxes_ = bboxes_.normalize()
        assert normalized_bboxes_.to_tensor().eq(bboxes).all()
        assert normalized_bboxes_.normalized
        assert not normalized_bboxes_.has_image_wh

        bboxes_ = PseudoBBoxes(bboxes, image_wh=(10, 20))
        normalized_bboxes = torch.tensor([[1.0, 1.0, 3.0, 2.5]])
        normalized_bboxes_ = bboxes_.normalize()
        assert normalized_bboxes_.to_tensor().eq(normalized_bboxes).all()
        assert normalized_bboxes_.normalized
        assert normalized_bboxes_.image_wh == (10, 20)

        bboxes_ = PseudoBBoxes(bboxes, normalized=True, image_wh=(10, 20))
        normalized_bboxes_ = bboxes_.normalize()
        assert normalized_bboxes_.to_tensor().eq(bboxes).all()
        assert normalized_bboxes_.normalized
        assert normalized_bboxes_.image_wh == (10, 20)

    def test_denormalize(self) -> None:
        bboxes = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_ = PseudoBBoxes(bboxes)
        denormalized_bboxes_ = bboxes_.denormalize()
        assert denormalized_bboxes_.to_tensor().eq(bboxes).all()
        assert not denormalized_bboxes_.normalized
        assert not denormalized_bboxes_.has_image_wh

        bboxes_ = PseudoBBoxes(bboxes, normalized=True)
        with pytest.raises(AttributeError):
            bboxes_.denormalize()

        bboxes_ = PseudoBBoxes(bboxes, image_wh=(10, 20))
        denormalized_bboxes_ = bboxes_.denormalize()
        assert denormalized_bboxes_.to_tensor().eq(bboxes).all()
        assert not denormalized_bboxes_.normalized
        assert denormalized_bboxes_.image_wh == (10, 20)

        bboxes_ = PseudoBBoxes(bboxes, normalized=True, image_wh=(10, 20))
        denormalized_bboxes = torch.tensor([[100.0, 400.0, 300.0, 1000.0]])
        denormalized_bboxes_ = bboxes_.denormalize()
        assert denormalized_bboxes_.to_tensor().eq(denormalized_bboxes).all()
        assert not denormalized_bboxes_.normalized
        assert denormalized_bboxes_.image_wh == (10, 20)


class PseudoBBoxesXY__(BBoxesXY__, PseudoBBoxes):
    pass


class TestBBoxesXY__:

    @pytest.fixture(scope='class')
    def bboxes_(self) -> PseudoBBoxesXY__:
        bboxes = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_ = PseudoBBoxesXY__(bboxes)
        return bboxes_

    def test_left(self, bboxes_: PseudoBBoxesXY__) -> None:
        assert bboxes_.left.item() == 10.0

    def test_top(self, bboxes_: PseudoBBoxesXY__) -> None:
        assert bboxes_.top.item() == 20.0

    def test_lt(self, bboxes_: PseudoBBoxesXY__) -> None:
        assert bboxes_.lt.tolist() == [[10.0, 20.0]]

    def test_from1(self, bboxes_: PseudoBBoxesXY__) -> None:
        assert PseudoBBoxesXY__._from1(bboxes_).tolist() == [[10.0, 20.0]]


class PseudoBBoxesCXCY__(BBoxesCXCY__, PseudoBBoxes):
    pass


class TestBBoxesCXCY__:

    @pytest.fixture(scope='class')
    def bboxes_(self) -> PseudoBBoxesCXCY__:
        bboxes = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_ = PseudoBBoxesCXCY__(bboxes)
        return bboxes_

    def test_center_x(self, bboxes_: PseudoBBoxesCXCY__) -> None:
        assert bboxes_.center_x.item() == 10.0

    def test_center_y(self, bboxes_: PseudoBBoxesCXCY__) -> None:
        assert bboxes_.center_y.item() == 20.0

    def test_center(self, bboxes_: PseudoBBoxesCXCY__) -> None:
        assert bboxes_.center.tolist() == [[10.0, 20.0]]

    def test_from1(self, bboxes_: PseudoBBoxesCXCY__) -> None:
        assert PseudoBBoxesCXCY__._from1(bboxes_).tolist() == [[10.0, 20.0]]


class PseudoBBoxes__XY(BBoxes__XY, PseudoBBoxes):  # noqa: N801
    pass


class TestBBoxes__XY:  # noqa: N801

    @pytest.fixture(scope='class')
    def bboxes_(self) -> PseudoBBoxes__XY:
        bboxes = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_ = PseudoBBoxes__XY(bboxes)
        return bboxes_

    def test_right(self, bboxes_: PseudoBBoxes__XY) -> None:
        assert bboxes_.right.item() == 30.0

    def test_bottom(self, bboxes_: PseudoBBoxes__XY) -> None:
        assert bboxes_.bottom.item() == 50.0

    def test_rb(self, bboxes_: PseudoBBoxes__XY) -> None:
        assert bboxes_.rb.tolist() == [[30.0, 50.0]]

    def test_from2(self, bboxes_: PseudoBBoxes__XY) -> None:
        assert PseudoBBoxes__XY._from2(bboxes_).tolist() == [[30.0, 50.0]]


class PseudoBBoxes__WH(BBoxes__WH, PseudoBBoxes):  # noqa: N801
    pass


class TestBBoxes__WH:  # noqa: N801

    @pytest.fixture(scope='class')
    def bboxes_(self) -> PseudoBBoxes__WH:
        bboxes = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_ = PseudoBBoxes__WH(bboxes)
        return bboxes_

    def test_width(self, bboxes_: PseudoBBoxes__WH) -> None:
        assert bboxes_.width.item() == 30.0

    def test_height(self, bboxes_: PseudoBBoxes__WH) -> None:
        assert bboxes_.height.item() == 50.0

    def test_wh(self, bboxes_: PseudoBBoxes__WH) -> None:
        assert bboxes_.wh.tolist() == [[30.0, 50.0]]

    def test_from2(self, bboxes_: PseudoBBoxes__WH) -> None:
        assert PseudoBBoxes__WH._from2(bboxes_).tolist() == [[30.0, 50.0]]


class TestBBoxesXYXY:

    @pytest.fixture(scope='class')
    def bboxes_(self) -> BBoxesXYXY:
        bboxes = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_ = BBoxesXYXY(bboxes)
        return bboxes_

    def test_width(self, bboxes_: BBoxesXYXY) -> None:
        assert bboxes_.width.item() == 20.0

    def test_height(self, bboxes_: BBoxesXYXY) -> None:
        assert bboxes_.height.item() == 30.0

    def test_center_x(self, bboxes_: BBoxesXYXY) -> None:
        assert bboxes_.center_x.item() == 20.0

    def test_center_y(self, bboxes_: BBoxesXYXY) -> None:
        assert bboxes_.center_y.item() == 35.0

    def test_wh(self, bboxes_: BBoxesXYXY) -> None:
        assert bboxes_.wh.tolist() == [[20.0, 30.0]]

    def test_center(self, bboxes_: BBoxesXYXY) -> None:
        assert bboxes_.center.tolist() == [[20.0, 35.0]]

    def test_round(self) -> None:
        bboxes = torch.tensor([[10.9, 20.3, 39.2, 99.8]])
        bboxes_ = BBoxesXYXY(bboxes)
        rounded_bboxes = torch.tensor([10.0, 20.0, 40.0, 100.0])
        rounded_bboxes_ = bboxes_.round()
        assert rounded_bboxes_.to_tensor().eq(rounded_bboxes).all()
        assert not rounded_bboxes_.normalized
        assert not rounded_bboxes_.has_image_wh

        bboxes_ = BBoxesXYXY(bboxes, image_wh=(50, 100))
        rounded_bboxes_ = bboxes_.round()
        assert rounded_bboxes_.to_tensor().eq(rounded_bboxes).all()
        assert not rounded_bboxes_.normalized
        assert rounded_bboxes_.image_wh == (50, 100)

        bboxes = torch.tensor([[0.218, 0.203, 0.784, 0.998]])
        bboxes_ = BBoxesXYXY(bboxes, normalized=True)
        with pytest.raises(AttributeError):
            bboxes_.round()

        bboxes_ = BBoxesXYXY(bboxes, normalized=True, image_wh=(50, 100))
        rounded_bboxes = torch.tensor([0.2, 0.2, 0.8, 1.0])
        rounded_bboxes_ = bboxes_.round()
        assert rounded_bboxes_.to_tensor().allclose(rounded_bboxes)
        assert rounded_bboxes_.normalized
        assert rounded_bboxes_.image_wh == (50, 100)

    def test_clamp(self) -> None:
        bboxes = torch.tensor([[10, 20, 30, 50]])
        bboxes_ = BBoxesXYXY(bboxes)
        with pytest.raises(AttributeError):
            bboxes_.clamp()

        bboxes_ = BBoxesXYXY(bboxes, normalized=True)
        with pytest.raises(AttributeError):
            bboxes_.clamp()

        bboxes_ = BBoxesXYXY(bboxes, image_wh=(50, 100))
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(bboxes).all()
        assert not clamped_bboxes_.normalized
        assert clamped_bboxes_.image_wh == (50, 100)

        bboxes_ = BBoxesXYXY(bboxes, image_wh=(30, 50))
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(bboxes).all()

        bboxes_ = BBoxesXYXY(bboxes, image_wh=(20, 30))
        clamped_bboxes = torch.tensor([[10, 20, 20, 30]])
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(clamped_bboxes).all()

        bboxes_ = BBoxesXYXY(bboxes, image_wh=(10, 20))
        clamped_bboxes = torch.tensor([[10, 20, 10, 20]])
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(clamped_bboxes).all()

        bboxes_ = BBoxesXYXY(bboxes, image_wh=(5, 10))
        clamped_bboxes = torch.tensor([[5, 10, 5, 10]])
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(clamped_bboxes).all()

        bboxes = torch.tensor([[-10, -20, 30, 50]])
        bboxes_ = BBoxesXYXY(bboxes, image_wh=(5, 10))
        clamped_bboxes = torch.tensor([[0, 0, 5, 10]])
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(clamped_bboxes).all()

    def test_area(self, bboxes_: BBoxesXYXY) -> None:
        assert bboxes_.area.item() == 600.0

    def test_from(self) -> None:
        bboxes_xyxy = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_xyxy_ = BBoxesXYXY(bboxes_xyxy)
        bboxes_ = BBoxesXYXY.from_(bboxes_xyxy_)
        assert bboxes_.to_tensor().eq(bboxes_xyxy).all()
        assert not bboxes_.normalized
        assert not bboxes_.has_image_wh

        bboxes_xyxy_ = BBoxesXYXY(bboxes_xyxy, normalized=True)
        bboxes_ = BBoxesXYXY.from_(bboxes_xyxy_)
        assert bboxes_.to_tensor().eq(bboxes_xyxy).all()
        assert bboxes_.normalized
        assert not bboxes_.has_image_wh

        bboxes_xyxy_ = BBoxesXYXY(bboxes_xyxy, image_wh=(10, 20))
        bboxes_ = BBoxesXYXY.from_(bboxes_xyxy_)
        assert bboxes_.to_tensor().eq(bboxes_xyxy).all()
        assert not bboxes_.normalized
        assert bboxes_.image_wh == (10, 20)

        bboxes_xyxy_ = BBoxesXYXY(
            bboxes_xyxy,
            normalized=True,
            image_wh=(10, 20),
        )
        bboxes_ = BBoxesXYXY.from_(bboxes_xyxy_)
        assert bboxes_.to_tensor().eq(bboxes_xyxy).all()
        assert bboxes_.normalized
        assert bboxes_.image_wh == (10, 20)

        bboxes_xywh = torch.tensor([[10.0, 20.0, 20.0, 30.0]])
        bboxes_xywh_ = BBoxesXYWH(bboxes_xywh)
        bboxes_ = BBoxesXYXY.from_(bboxes_xywh_)
        assert bboxes_.to_tensor().eq(bboxes_xyxy).all()

        bboxes_cxcywh = torch.tensor([[20.0, 35.0, 20.0, 30.0]])
        bboxes_cxcywh_ = BBoxesCXCYWH(bboxes_cxcywh)
        bboxes_ = BBoxesXYXY.from_(bboxes_cxcywh_)
        assert bboxes_.to_tensor().eq(bboxes_xyxy).all()

    def test_to(self) -> None:
        bboxes_xyxy = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        bboxes_xyxy_ = BBoxesXYXY(bboxes_xyxy)
        bboxes_ = bboxes_xyxy_.to(BBoxesXYXY)
        assert bboxes_.to_tensor().eq(bboxes_xyxy).all()
        assert not bboxes_.normalized
        assert not bboxes_.has_image_wh

        bboxes_xyxy_ = BBoxesXYXY(bboxes_xyxy, normalized=True)
        bboxes_ = bboxes_xyxy_.to(BBoxesXYXY)
        assert bboxes_.to_tensor().eq(bboxes_xyxy).all()
        assert bboxes_.normalized
        assert not bboxes_.has_image_wh

        bboxes_xyxy_ = BBoxesXYXY(bboxes_xyxy, image_wh=(10, 20))
        bboxes_ = bboxes_xyxy_.to(BBoxesXYXY)
        assert bboxes_.to_tensor().eq(bboxes_xyxy).all()
        assert not bboxes_.normalized
        assert bboxes_.image_wh == (10, 20)

        bboxes_xyxy_ = BBoxesXYXY(
            bboxes_xyxy,
            normalized=True,
            image_wh=(10, 20),
        )
        bboxes_ = bboxes_xyxy_.to(BBoxesXYXY)
        assert bboxes_.to_tensor().eq(bboxes_xyxy).all()
        assert bboxes_.normalized
        assert bboxes_.image_wh == (10, 20)

        bboxes_xywh = torch.tensor([[10.0, 20.0, 20.0, 30.0]])
        bboxes_xywh_ = BBoxesXYWH(bboxes_xywh)
        bboxes_ = bboxes_xywh_.to(BBoxesXYXY)
        assert bboxes_.to_tensor().eq(bboxes_xyxy).all()

        bboxes_cxcywh = torch.tensor([[20.0, 35.0, 20.0, 30.0]])
        bboxes_cxcywh_ = BBoxesCXCYWH(bboxes_cxcywh)
        bboxes_ = bboxes_cxcywh_.to(BBoxesXYXY)
        assert bboxes_.to_tensor().eq(bboxes_xyxy).all()

    def test_ious(self) -> None:
        bboxes1 = torch.tensor([
            [10.0, 20.0, 40.0, 100.0],
        ])
        bboxes2 = torch.tensor([
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

        bboxes1_ = BBoxesXYXY(bboxes1)
        bboxes2_ = BBoxesXYXY(bboxes2)

        intersections = torch.tensor([[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 600.0, 1200.0, 300.0, 600.0,
            0.0, 0.0, 1200.0, 2400.0, 600.0, 1200.0, 0.0, 0.0, 400.0, 800.0,
            200.0, 400.0, 0.0, 0.0, 600.0, 1200.0, 300.0, 600.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ]])
        intersections_ = bboxes1_.intersections(bboxes2_)
        assert intersections_.eq(intersections).all()
        intersections_ = bboxes1_ & bboxes2_
        assert intersections_.eq(intersections).all()

        unions = torch.tensor([[
            2409.0, 2535.0, 2670.0, 2460.0, 2535.0, 2409.0, 2460.0, 2700.0,
            3000.0, 2500.0, 2700.0, 2460.0, 2520.0, 3000.0, 3600.0, 2600.0,
            3000.0, 2520.0, 2430.0, 2450.0, 2500.0, 2400.0, 2450.0, 2430.0,
            2460.0, 2700.0, 3000.0, 2500.0, 2700.0, 2460.0, 2409.0, 2535.0,
            2670.0, 2460.0, 2535.0, 2409.0
        ]])
        unions_ = bboxes1_.unions(bboxes2_, intersections)
        assert unions_.eq(unions).all()
        unions_ = bboxes1_ | bboxes2_
        assert unions_.eq(unions).all()

        ious = intersections / unions
        ious_ = bboxes1_.ious(bboxes2_)
        assert ious_.eq(ious).all()

    def test_expand(self, bboxes_: BBoxesXYXY) -> None:
        expanded_bboxes = torch.tensor([[0.0, 5.0, 40.0, 65.0]])
        expanded_bboxes_ = bboxes_.expand((2, 2))
        assert expanded_bboxes_.to_tensor().eq(expanded_bboxes).all()

    def test_indices(self, bboxes_: BBoxesXYXY) -> None:
        assert bboxes_.indices().item()
        assert bboxes_.indices(min_area=600.0).item()
        assert not bboxes_.indices(min_area=601.0).item()
        assert bboxes_.indices(min_wh=(20.0, 30.0)).item()
        assert not bboxes_.indices(min_wh=(21.0, 30.0)).item()
        assert not bboxes_.indices(min_wh=(20.0, 31.0)).item()
        assert not bboxes_.indices(min_wh=(21.0, 31.0)).item()


class TestBBoxesXYWH:

    @pytest.fixture(scope='class')
    def bboxes_(self) -> BBoxesXYWH:
        bboxes = torch.tensor([[10.0, 20.0, 20.0, 30.0]])
        bboxes_ = BBoxesXYWH(bboxes)
        return bboxes_

    def test_right(self, bboxes_: BBoxesXYWH) -> None:
        assert bboxes_.right.item() == 30.0

    def test_bottom(self, bboxes_: BBoxesXYWH) -> None:
        assert bboxes_.bottom.item() == 50.0

    def test_center_x(self, bboxes_: BBoxesXYWH) -> None:
        assert bboxes_.center_x.item() == 20.0

    def test_center_y(self, bboxes_: BBoxesXYWH) -> None:
        assert bboxes_.center_y.item() == 35.0

    def test_rb(self, bboxes_: BBoxesXYWH) -> None:
        assert bboxes_.rb.tolist() == [[30.0, 50.0]]

    def test_center(self, bboxes_: BBoxesXYWH) -> None:
        assert bboxes_.center.tolist() == [[20.0, 35.0]]

    def test_round(self) -> None:
        bboxes = torch.tensor([[10.9, 20.3, 28.3, 79.5]])
        bboxes_ = BBoxesXYWH(bboxes)
        rounded_bboxes = torch.tensor([10.0, 20.0, 30.0, 80.0])
        rounded_bboxes_ = bboxes_.round()
        assert rounded_bboxes_.to_tensor().eq(rounded_bboxes).all()
        assert not rounded_bboxes_.normalized
        assert not rounded_bboxes_.has_image_wh

        bboxes_ = BBoxesXYWH(bboxes, image_wh=(50, 100))
        rounded_bboxes_ = bboxes_.round()
        assert rounded_bboxes_.to_tensor().eq(rounded_bboxes).all()
        assert not rounded_bboxes_.normalized
        assert rounded_bboxes_.image_wh == (50, 100)

        bboxes = torch.tensor([[0.218, 0.203, 0.566, 0.795]])
        bboxes_ = BBoxesXYWH(bboxes, normalized=True)
        with pytest.raises(AttributeError):
            bboxes_.round()

        bboxes_ = BBoxesXYWH(bboxes, normalized=True, image_wh=(50, 100))
        rounded_bboxes = torch.tensor([0.2, 0.2, 0.6, 0.8])
        rounded_bboxes_ = bboxes_.round()
        assert rounded_bboxes_.to_tensor().allclose(rounded_bboxes)
        assert rounded_bboxes_.normalized
        assert rounded_bboxes_.image_wh == (50, 100)

    def test_expand(self, bboxes_: BBoxesXYWH) -> None:
        expanded_bboxes = torch.tensor([[0.0, 5.0, 40.0, 60.0]])
        expanded_bboxes_ = bboxes_.expand((2, 2))
        assert expanded_bboxes_.to_tensor().eq(expanded_bboxes).all()

    def test_clamp(self) -> None:
        bboxes = torch.tensor([[10, 20, 20, 30]])
        bboxes_ = BBoxesXYWH(bboxes)
        with pytest.raises(AttributeError):
            bboxes_.clamp()

        bboxes_ = BBoxesXYWH(bboxes, normalized=True)
        with pytest.raises(AttributeError):
            bboxes_.clamp()

        bboxes_ = BBoxesXYWH(bboxes, image_wh=(50, 100))
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(bboxes).all()
        assert not clamped_bboxes_.normalized
        assert clamped_bboxes_.image_wh == (50, 100)

        bboxes_ = BBoxesXYWH(bboxes, image_wh=(30, 50))
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(bboxes).all()

        bboxes_ = BBoxesXYWH(bboxes, image_wh=(20, 30))
        clamped_bboxes = torch.tensor([[10, 20, 10, 10]])
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(clamped_bboxes).all()

        bboxes_ = BBoxesXYWH(bboxes, image_wh=(10, 20))
        clamped_bboxes = torch.tensor([[10, 20, 0, 0]])
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(clamped_bboxes).all()

        bboxes_ = BBoxesXYWH(bboxes, image_wh=(5, 10))
        clamped_bboxes = torch.tensor([[5, 10, 0, 0]])
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(clamped_bboxes).all()

        bboxes = torch.tensor([[-10, -20, 40, 70]])
        bboxes_ = BBoxesXYWH(bboxes, image_wh=(5, 10))
        clamped_bboxes = torch.tensor([[0, 0, 5, 10]])
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(clamped_bboxes).all()


class TestBBBoxesCXCYWH:

    @pytest.fixture(scope='class')
    def bboxes_(self) -> BBoxesCXCYWH:
        bboxes = torch.tensor([[20.0, 35.0, 20.0, 30.0]])
        bboxes_ = BBoxesCXCYWH(bboxes)
        return bboxes_

    def test_left(self, bboxes_: BBoxesCXCYWH) -> None:
        assert bboxes_.left.item() == 10.0

    def test_right(self, bboxes_: BBoxesCXCYWH) -> None:
        assert bboxes_.right.item() == 30.0

    def test_top(self, bboxes_: BBoxesCXCYWH) -> None:
        assert bboxes_.top.item() == 20.0

    def test_bottom(self, bboxes_: BBoxesCXCYWH) -> None:
        assert bboxes_.bottom.item() == 50.0

    def test_lt(self, bboxes_: BBoxesCXCYWH) -> None:
        assert bboxes_.lt.tolist() == [[10.0, 20.0]]

    def test_rb(self, bboxes_: BBoxesCXCYWH) -> None:
        assert bboxes_.rb.tolist() == [[30.0, 50.0]]

    def test_expand(self, bboxes_: BBoxesCXCYWH) -> None:
        expanded_bboxes = torch.tensor([[20.0, 35.0, 40.0, 60.0]])
        expanded_bboxes_ = bboxes_.expand((2, 2))
        assert expanded_bboxes_.to_tensor().eq(expanded_bboxes).all()

    def test_round(self) -> None:
        bboxes = torch.tensor([[25.05, 60.05, 28.3, 79.5]])
        bboxes_ = BBoxesCXCYWH(bboxes)
        rounded_bboxes = torch.tensor([25.0, 60.0, 30.0, 80.0])
        rounded_bboxes_ = bboxes_.round()
        assert rounded_bboxes_.to_tensor().eq(rounded_bboxes).all()
        assert not rounded_bboxes_.normalized
        assert not rounded_bboxes_.has_image_wh

        bboxes_ = BBoxesCXCYWH(bboxes, image_wh=(50, 100))
        rounded_bboxes_ = bboxes_.round()
        assert rounded_bboxes_.to_tensor().eq(rounded_bboxes).all()
        assert not rounded_bboxes_.normalized
        assert rounded_bboxes_.image_wh == (50, 100)

        bboxes = torch.tensor([[0.501, 0.6005, 0.566, 0.795]])
        bboxes_ = BBoxesCXCYWH(bboxes, normalized=True)
        with pytest.raises(AttributeError):
            bboxes_.round()

        bboxes_ = BBoxesCXCYWH(bboxes, normalized=True, image_wh=(50, 100))
        rounded_bboxes = torch.tensor([0.5, 0.6, 0.6, 0.8])
        rounded_bboxes_ = bboxes_.round()
        assert rounded_bboxes_.to_tensor().allclose(rounded_bboxes)
        assert rounded_bboxes_.normalized
        assert rounded_bboxes_.image_wh == (50, 100)

    def test_clamp(self) -> None:
        bboxes = torch.tensor([[20, 35, 20, 30]])
        bboxes_ = BBoxesCXCYWH(bboxes)
        with pytest.raises(AttributeError):
            bboxes_.clamp()

        bboxes_ = BBoxesCXCYWH(bboxes, normalized=True)
        with pytest.raises(AttributeError):
            bboxes_.clamp()

        bboxes_ = BBoxesCXCYWH(bboxes, image_wh=(50, 100))
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(bboxes).all()
        assert not clamped_bboxes_.normalized
        assert clamped_bboxes_.image_wh == (50, 100)

        bboxes_ = BBoxesCXCYWH(bboxes, image_wh=(30, 50))
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(bboxes).all()

        bboxes_ = BBoxesCXCYWH(bboxes, image_wh=(20, 30))
        clamped_bboxes = torch.tensor([[15, 25, 10, 10]])
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(clamped_bboxes).all()

        bboxes_ = BBoxesCXCYWH(bboxes, image_wh=(10, 20))
        clamped_bboxes = torch.tensor([[10, 20, 0, 0]])
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(clamped_bboxes).all()

        bboxes_ = BBoxesCXCYWH(bboxes, image_wh=(5, 10))
        clamped_bboxes = torch.tensor([[5, 10, 0, 0]])
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(clamped_bboxes).all()

        bboxes = torch.tensor([[10, 15, 40, 70]])
        bboxes_ = BBoxesCXCYWH(bboxes, image_wh=(5, 10))
        clamped_bboxes = torch.tensor([[2.5, 5, 5, 10]])
        clamped_bboxes_ = bboxes_.clamp()
        assert clamped_bboxes_.to_tensor().eq(clamped_bboxes).all()
