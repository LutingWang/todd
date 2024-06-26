import pathlib

import torch

import todd.tasks.knowledge_distillation as kd
from todd.configs import PyConfig
from todd.utils import NestedTensorCollectionUtils

BaseDistiller = kd.distillers.BaseDistiller
DistillerStore = kd.distillers.DistillerStore
KDAdaptRegistry = kd.distillers.KDAdaptRegistry
BaseAdapt = kd.distillers.adapts.BaseAdapt


@KDAdaptRegistry.register_()
class CustomAdapt(BaseAdapt):

    def __init__(self, stride: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stride = stride

    def forward(
        self,
        feat: torch.Tensor,
        bboxes: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        valid_idx = pos[:, 1] >= 0
        feat = feat[valid_idx]
        bboxes = bboxes[valid_idx]
        pos = pos[valid_idx]
        bs, level, h, w, id_ = pos.split(1, 1)
        bbox_list = [bboxes[bs.flatten() == i] for i in range(2)]
        h = torch.div(h, self._stride, rounding_mode='trunc')
        w = torch.div(w, self._stride, rounding_mode='trunc')
        pos = torch.cat((level, bs, h, w), dim=-1)
        id_ = id_.reshape(-1)
        return feat, bbox_list, pos, id_


class TestCKD:

    # @pytest.fixture
    # def adaptflow(self) -> Workflow:
    #     return Workflow.build(
    #         'AdaptRegistry', {
    #             'pred_reshaped': dict(
    #                 type='Rearrange',
    #                 fields=['preds'],
    #                 parallel=True,
    #                 pattern='bs dim h w -> bs h w dim',
    #             ),
    #             ('targets', 'bboxes', 'bbox_poses', 'anchor_ids'): dict(
    #                 type='CustomAdapt',
    #                 fields=['targets', 'bboxes', 'bbox_ids'],
    #                 stride=1,
    #             ),
    #             'pred_indexed': dict(
    #                 type='Index',
    #                 fields=['pred_reshaped', 'bbox_poses'],
    #             ),
    #             'preds': dict(
    #                 type='Decouple',
    #                 fields=['pred_indexed', 'anchor_ids'],
    #                 num=9,
    #                 in_features=4,
    #                 out_features=16,
    #                 bias=False,
    #             ),
    #         }
    #     )

    # @pytest.fixture
    # def ckdflow(self) -> Workflow:
    #     return Workflow.build(
    #         'LossRegistry',
    #         loss_ckd=dict(
    #             type='CKDLoss',
    #             fields=['preds', 'targets', 'bboxes'],
    #             weight=0.5,
    #         ),
    #     )

    # @pytest.fixture(scope='class')
    # def result(self) -> dict[str, dict]:
    #     filename = os.path.join(os.path.dirname(__file__), 'ckd.pth')
    #     return torch.load(filename, map_location='cpu')

    # @torch.no_grad()
    # def generate_result(self) -> None:
    #     from todd.utils import CollectionTensor
    #     ckd = self.ckd()
    #     old_filename = os.path.join(os.path.dirname(__file__), 'ckd.pth')
    #     filename = os.path.join(os.path.dirname(__file__), 'ckd.pth.tmp')
    #     old_result: dict = torch.load(old_filename, map_location='cpu')
    #     result = {}
    #     for rank, old_rank_result in old_result.items():
    #         adapt = self.adapt()
    #         old_inputs = old_rank_result['inputs']
    #         inputs = {
    #             'preds': [f[:, :4, :, :] for f in old_inputs['preds']],
    #             'targets': old_inputs['targets'][:, :16],
    #             'bboxes': old_inputs['bboxes'],
    #             'bbox_ids': old_inputs['bbox_ids'],
    #         }
    #         tensors = adapt(inputs, inplace=False)
    #         losses = ckd(tensors)
    #         result[rank] = CollectionTensor.apply(
    #             {
    #                 'state_dict': adapt.state_dict(),
    #                 'inputs': inputs,
    #                 'tensors': tensors,
    #                 'losses': losses,
    #             },
    #             lambda f: f.contiguous(),
    #         )
    #     torch.save(result, filename)

    def test_ckd(self, data_dir: pathlib.Path) -> None:
        config = PyConfig.load(data_dir / 'ckd.py')
        result = torch.load(data_dir / 'ckd.pth', map_location='cpu')

        distiller = BaseDistiller(**config.distiller)

        for rank in range(4):
            rank_result = result[f'rank{rank}']
            distiller.load_state_dict(rank_result['state_dict'])

            assert not DistillerStore.INTERMEDIATE_OUTPUTS
            DistillerStore.INTERMEDIATE_OUTPUTS = '_debug_'
            losses = distiller(rank_result['inputs'])
            DistillerStore.INTERMEDIATE_OUTPUTS = ''
            tensors = losses.pop('_debug_')
            tensors.pop('bbox_ids')

            utils = NestedTensorCollectionUtils()
            assert utils.all_close(rank_result['tensors'], tensors)
            assert utils.all_close(rank_result['losses'], losses)


# if __name__ == '__main__':
#     test = TestCKD()
#     test.generate_result()
