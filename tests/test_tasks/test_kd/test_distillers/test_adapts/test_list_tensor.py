import pytest
import torch

from todd.utils import TensorTreeUtil, TreeUtil


@pytest.fixture(scope='module')
def tensor_feat() -> torch.Tensor:
    return torch.arange(720).reshape(6, 5, 4, 3, 2)


# yapf: disable
@pytest.fixture(scope='module')
def list_feat() -> list[list[list[torch.Tensor]]]:
    return [
        [[torch.arange(6).reshape(3, 2) + a * 120 + b * 24 + c * 6
          for c in range(4)]
         for b in range(5)]
        for a in range(6)
    ]


@pytest.fixture(scope='module')
def hybrid_feat() -> list:
    return [
        torch.arange(120).reshape(5, 4, 3, 2),
        [torch.arange(24).reshape(4, 3, 2) + 120 + b * 24
         for b in range(5)],
        [[torch.arange(6).reshape(3, 2) + 240 + b * 24 + c * 6
          for c in range(4)]
         for b in range(5)],
        [[[torch.arange(2) + 360 + b * 24 + c * 6 + d * 2
           for d in range(3)]
          for c in range(4)]
         for b in range(5)],
        [[[[torch.tensor(480 + b * 24 + c * 6 + d * 2 + e)
            for e in range(2)]
           for d in range(3)]
          for c in range(4)]
         for b in range(5)],
        [torch.arange(24).reshape(4, 3, 2) + 600,
         [torch.arange(6).reshape(3, 2) + 624 + c * 6
          for c in range(4)],
         [[torch.arange(2) + 648 + c * 6 + d * 2
           for d in range(3)]
          for c in range(4)],
         [[[torch.tensor(672 + c * 6 + d * 2 + e)
            for e in range(2)]
           for d in range(3)]
          for c in range(4)],
         [torch.arange(6).reshape(3, 2) + 696,
          [torch.arange(2) + 702 + d * 2 for d in range(3)],
          [[torch.tensor(708 + d * 2 + e)
            for e in range(2)]
           for d in range(3)],
          [torch.arange(2) + 714,
           [torch.tensor(716 + e) for e in range(2)],
           [torch.tensor(718 + e) for e in range(2)]]]],
    ]
# yapf: enable


class TestStack:

    @pytest.mark.parametrize(
        'feat',
        ['tensor_feat', 'list_feat', 'hybrid_feat'],
    )
    def test_normal(self, feat: str, request: pytest.FixtureRequest) -> None:
        feat_ = request.getfixturevalue(feat)
        stacked_feat: torch.Tensor = TreeUtil.reduce(torch.stack, feat_)
        stacked_feat = stacked_feat.reshape(-1)
        # stacked_feat = ListTensor.stack(feat_).reshape(-1)
        assert torch.arange(720).eq(stacked_feat).all()


class TestIndex:

    @pytest.mark.parametrize(
        'feat',
        ['tensor_feat', 'list_feat', 'hybrid_feat'],
    )
    def test_empty_pos(
        self, feat: str, request: pytest.FixtureRequest
    ) -> None:
        feat_ = request.getfixturevalue(feat)
        indexed_feat = TensorTreeUtil.index(feat_, torch.zeros([0, 0]))
        assert indexed_feat.shape == (0, 6, 5, 4, 3, 2)
        indexed_feat = TensorTreeUtil.index(feat_, torch.zeros([0, 1]))
        assert indexed_feat.shape == (0, 5, 4, 3, 2)
        indexed_feat = TensorTreeUtil.index(feat_, torch.zeros([0, 2]))
        assert indexed_feat.shape == (0, 4, 3, 2)
        indexed_feat = TensorTreeUtil.index(feat_, torch.zeros([0, 3]))
        assert indexed_feat.shape == (0, 3, 2)
        indexed_feat = TensorTreeUtil.index(feat_, torch.zeros([0, 4]))
        assert indexed_feat.shape == (0, 2)
        indexed_feat = TensorTreeUtil.index(feat_, torch.zeros([100, 0]))
        assert indexed_feat.shape == (100, 6, 5, 4, 3, 2)

    @pytest.mark.parametrize(
        'feat',
        ['tensor_feat', 'list_feat', 'hybrid_feat'],
    )
    def test_normal(self, feat: str, request: pytest.FixtureRequest) -> None:
        feat_ = request.getfixturevalue(feat)
        pos = torch.Tensor([[0, 1], [1, 0], [1, 2]])
        indexed_feat = TensorTreeUtil.index(feat_, pos)
        assert torch.stack([
            torch.arange(24, 48).reshape(4, 3, 2),
            torch.arange(120, 144).reshape(4, 3, 2),
            torch.arange(168, 192).reshape(4, 3, 2),
        ]).eq(indexed_feat).all()
