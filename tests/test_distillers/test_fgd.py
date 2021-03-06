import os
from typing import Dict

import pytest
import torch

from todd.distillers.base import BaseDistiller
from todd.utils import CollectionTensor


class TestFGD:

    @pytest.fixture()
    def distiller(self):
        distiller = BaseDistiller(
            models=list(),
            hooks=dict(),
            adapts={
                'attn_spatial': dict(
                    type='AbsMeanSpatialAttention',
                    fields=['neck'],
                    parallel=True,
                    temperature=0.5,
                ),
                'teacher_attn_spatial': dict(
                    type='AbsMeanSpatialAttention',
                    fields=['teacher_neck'],
                    parallel=True,
                    temperature=0.5,
                ),
                'attn_channel': dict(
                    type='AbsMeanChannelAttention',
                    fields=['neck'],
                    parallel=True,
                    temperature=0.5,
                ),
                'teacher_attn_channel': dict(
                    type='AbsMeanChannelAttention',
                    fields=['teacher_neck'],
                    parallel=True,
                    temperature=0.5,
                ),
                'masks': dict(
                    type='FGDMask',
                    fields=['img_shape', 'gt_bboxes'],
                    neg_gain=0.5,
                    strides=[8, 16, 32, 64, 128],
                    ceil_mode=True,
                ),
                'global': dict(
                    type='ContextBlock',
                    fields=['neck'],
                    parallel=5,
                    in_channels=4,
                    ratio=0.5,
                ),
                'teacher_global': dict(
                    type='ContextBlock',
                    fields=['teacher_neck'],
                    parallel=5,
                    in_channels=4,
                    ratio=0.5,
                ),
            },
            losses={
                'loss_feat': dict(
                    type='FGDLoss',
                    fields=[
                        'neck',
                        'teacher_neck',
                        'teacher_attn_spatial',
                        'teacher_attn_channel',
                        'masks',
                    ],
                    parallel=True,
                    weight=5e-4,
                    reduction='sum',
                ),
                'loss_attn_spatial': dict(
                    type='L1Loss',
                    fields=['attn_spatial', 'teacher_attn_spatial'],
                    parallel=True,
                    weight=2.5e-4,
                    reduction='sum',
                ),
                'loss_attn_channel': dict(
                    type='L1Loss',
                    fields=['attn_channel', 'teacher_attn_channel'],
                    parallel=True,
                    weight=2.5e-4,
                    reduction='sum',
                ),
                'loss_global': dict(
                    type='MSELoss',
                    fields=['global', 'teacher_global'],
                    parallel=True,
                    weight=2.5e-6,
                    reduction='sum',
                ),
            },
        )
        return distiller

    @pytest.fixture(scope='class')
    def result(self) -> Dict[str, dict]:
        filename = os.path.join(os.path.dirname(__file__), 'fgd.pth')
        return torch.load(filename, map_location='cpu')

    def tensors(self):
        tensors = {
            'neck': [
                torch.rand(2, 4, 112, 144),
                torch.rand(2, 4, 56, 72),
                torch.rand(2, 4, 28, 36),
                torch.rand(2, 4, 14, 18),
                torch.rand(2, 4, 7, 9),
            ],
            'teacher_neck': [
                torch.rand(2, 4, 112, 144),
                torch.rand(2, 4, 56, 72),
                torch.rand(2, 4, 28, 36),
                torch.rand(2, 4, 14, 18),
                torch.rand(2, 4, 7, 9),
            ],
            'gt_bboxes': [
                torch.Tensor([[10, 5, 22, 39]]),
                torch.Tensor([
                    [12, 9, 21, 19],
                    [15, 17, 29, 18],
                ]),
            ],
            'img_shape': (896, 1152),
        }
        return tensors

    @torch.no_grad()
    def generate_result(self):
        distiller = self.distiller()
        inputs = self.tensors()
        losses, tensors = distiller.distill(inputs, debug=True)
        result = {
            'state_dict': distiller.state_dict(),
            'inputs': inputs,
            'tensors': tensors,
            'losses': losses,
        }
        filename = os.path.join(os.path.dirname(__file__), 'fgd1.pth')
        torch.save(result, filename)

    @pytest.mark.usefixtures('setup_teardown_iter')
    @pytest.mark.parametrize(
        'setup_value,teardown_value',
        [(None, None)],
    )
    def test_fgd(self, distiller: BaseDistiller, result: Dict[str, dict]):
        distiller.load_state_dict(result['state_dict'])
        losses = distiller.distill(result['inputs'], debug=True, updated=True)
        tensors = {  # yapf: disable
            k[len(distiller.DEBUG_PREFIX):]: v
            for k, v in losses.items()
            if k.startswith(distiller.DEBUG_PREFIX)
        }
        for k in tensors:
            losses.pop(distiller.DEBUG_PREFIX + k)
        assert CollectionTensor.allclose(result['tensors'], tensors)
        assert CollectionTensor.allclose(result['losses'], losses)


if __name__ == '__main__':
    test = TestFGD()
    test.generate_result()
