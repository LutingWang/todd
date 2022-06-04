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
            models=None,
            hooks=None,
            trackings=None,
            adapts={
                'attn_spatial': dict(
                    type='AbsMeanSpatialAttention',
                    tensor_names=['neck'],
                    multilevel=True,
                    temperature=0.5,
                ),
                'teacher_attn_spatial': dict(
                    type='AbsMeanSpatialAttention',
                    tensor_names=['teacher_neck'],
                    multilevel=True,
                    temperature=0.5,
                ),
                'attn_channel': dict(
                    type='AbsMeanChannelAttention',
                    tensor_names=['neck'],
                    multilevel=True,
                    temperature=0.5,
                ),
                'teacher_attn_channel': dict(
                    type='AbsMeanChannelAttention',
                    tensor_names=['teacher_neck'],
                    multilevel=True,
                    temperature=0.5,
                ),
                'masks': dict(
                    type='FGDMask',
                    tensor_names=['img_shape', 'gt_bboxes'],
                    neg_gain=0.5,
                    strides=[8, 16, 32, 64, 128],
                    ceil_mode=True,
                ),
                'global': dict(
                    type='ContextBlock',
                    tensor_names=['neck'],
                    multilevel=5,
                    in_channels=4,
                    ratio=0.5,
                ),
                'teacher_global': dict(
                    type='ContextBlock',
                    tensor_names=['teacher_neck'],
                    multilevel=5,
                    in_channels=4,
                    ratio=0.5,
                ),
            },
            losses={
                'feat': dict(
                    type='FGDLoss',
                    tensor_names=[
                        'neck',
                        'teacher_neck',
                        'teacher_attn_spatial',
                        'teacher_attn_channel',
                        'masks',
                    ],
                    multilevel=True,
                    weight=5e-4,
                    reduction='sum',
                ),
                'attn_spatial': dict(
                    type='L1Loss',
                    tensor_names=['attn_spatial', 'teacher_attn_spatial'],
                    multilevel=True,
                    weight=2.5e-4,
                    reduction='sum',
                ),
                'attn_channel': dict(
                    type='L1Loss',
                    tensor_names=['attn_channel', 'teacher_attn_channel'],
                    multilevel=True,
                    weight=2.5e-4,
                    reduction='sum',
                ),
                'global': dict(
                    type='MSELoss',
                    tensor_names=['global', 'teacher_global'],
                    multilevel=True,
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

    def test_fgd(self, distiller: BaseDistiller, result: Dict[str, dict]):
        distiller.load_state_dict(result['state_dict'])
        losses, tensors = distiller.distill(result['inputs'], debug=True)
        assert CollectionTensor.allclose(result['tensors'], tensors)
        assert CollectionTensor.allclose(result['losses'], losses)


if __name__ == '__main__':
    test = TestFGD()
    test.generate_result()
