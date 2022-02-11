import os

import pytest
import torch

from todd.distillers.base import BaseDistiller


class TestFGD:
    @pytest.fixture()
    def distiller(self):
        distiller = BaseDistiller(
            models=None, hooks=None, trackings=None,
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
                ('masks', 'bg_masks'): dict(
                    type='FGDMask',
                    tensor_names=['img_shape', 'gt_bboxes'],
                    strides=[8, 16, 32, 64, 128],
                ),
                'attn_masks': dict(
                    type='Custom',
                    tensor_names=['teacher_attn_spatial', 'teacher_attn_channel', 'masks'],
                    multilevel=True,
                    pattern='a * b * c',
                ),
                'attn_bg_masks': dict(
                    type='Custom',
                    tensor_names=['teacher_attn_spatial', 'teacher_attn_channel', 'bg_masks'],
                    multilevel=True,
                    pattern='a * b * c',
                ),
                'global': dict(
                    type='ContextBlock',
                    tensor_names=['neck'],
                    multilevel=5,
                    in_channels=256,
                    ratio=0.5,
                ),
                'teacher_global': dict(
                    type='ContextBlock',
                    tensor_names=['teacher_neck'],
                    multilevel=5,
                    in_channels=256,
                    ratio=0.5,
                ),
            },
            losses={
                'feat': dict(
                    type='MSELoss',
                    tensor_names=['neck', 'teacher_neck', 'attn_masks'],
                    multilevel=True,
                    weight=5e-4,
                    reduction='sum',
                ),
                'bg_feat': dict(
                    type='MSELoss',
                    tensor_names=['neck', 'teacher_neck', 'attn_bg_masks'],
                    multilevel=True,
                    weight=2.5e-4,
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

    def tensors(self):
        tensors = {
            'neck': [
                torch.rand(2, 256, 112, 144),
                torch.rand(2, 256, 56, 72),
                torch.rand(2, 256, 28, 36),
                torch.rand(2, 256, 14, 18),
                torch.rand(2, 256, 7, 9),
            ],
            'teacher_neck': [
                torch.rand(2, 256, 112, 144),
                torch.rand(2, 256, 56, 72),
                torch.rand(2, 256, 28, 36),
                torch.rand(2, 256, 14, 18),
                torch.rand(2, 256, 7, 9),
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

    def generate_result(self):
        distiller = self.distiller()
        inputs = self.tensors()
        losses, tensors = distiller.distill(inputs, debug=True)
        result = {
            'state_dict': distiller.state_dict(),
            'inputs': inputs, 'tensors': tensors, 'losses': losses, 
        }
        filename = os.path.join(os.path.dirname(__file__), 'fgd.pth')
        torch.save(result, filename)

    def test_fgd(self, distiller: BaseDistiller):
        filename = os.path.join(os.path.dirname(__file__), 'fgd.pth')
        result = torch.load(filename)
        distiller.load_state_dict(result['state_dict'])
        losses, tensors = distiller.distill(result['inputs'], debug=True)
        assert result['tensors'].keys() == tensors.keys()
        for k in result['inputs']:
            result['tensors'].pop(k)
            tensors.pop(k)
        for k in result['tensors']:
            if isinstance(result['tensors'][k], torch.Tensor):
                assert isinstance(tensors[k], torch.Tensor), k
                assert torch.allclose(result['tensors'][k], tensors[k]), k
            else:
                for result_tensor, tensor in zip(result['tensors'][k], tensors[k]):
                    assert torch.allclose(result_tensor, tensor), k
        assert result['losses'].keys() == losses.keys()
        for k in result['losses']:
            if isinstance(result['losses'][k], torch.Tensor):
                assert isinstance(losses[k], torch.Tensor), k
                assert torch.allclose(result['losses'][k], losses[k]), k
            else:
                for result_tensor, tensor in zip(result['losses'][k], losses[k]):
                    assert torch.allclose(result_tensor, tensor), k


if __name__ == '__main__':
    test = TestFGD()
    test.test_fgd(test.distiller())