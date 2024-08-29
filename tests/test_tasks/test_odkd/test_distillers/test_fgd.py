import os
import pathlib

import pytest
import torch

import todd.tasks.knowledge_distillation as kd
from todd.configs import PyConfig
from todd.tasks.knowledge_distillation.distillers import (
    BaseDistiller,
    DistillerStore,
)
from todd.utils import NestedTensorCollectionUtils


class TestFGD:

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        from mmcv.cnn import ContextBlock

        kd.distillers.KDAdaptRegistry.register_(
            'mmcv_ContextBlock',
            force=True,
        )(ContextBlock)

    # @pytest.fixture
    # def distiller(self, data_dir: pathlib.Path) -> BaseDistiller:
    #     config = Config.load(data_dir / 'config.py')
    #     return BaseDistiller.build(config)

    # @pytest.fixture()
    # def result(self, data_dir: pathlib.Path):
    #     return torch.load(data_dir / 'fgd.pth', map_location='cpu')

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
        distiller = self.distiller()  # pylint: disable=no-member
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

    def test_fgd(self, data_dir: pathlib.Path) -> None:
        config = PyConfig.load(data_dir / 'fgd.py')
        result = torch.load(data_dir / 'fgd.pth', map_location='cpu')

        distiller = kd.KDDistillerRegistry.build(
            config.distiller,
            type=BaseDistiller.__name__,
        )
        distiller.load_state_dict(result['state_dict'])

        assert not DistillerStore.INTERMEDIATE_OUTPUTS
        DistillerStore.INTERMEDIATE_OUTPUTS = '_debug_'
        losses = distiller(result['inputs'])
        DistillerStore.INTERMEDIATE_OUTPUTS = ''
        tensors = losses.pop('_debug_')

        utils = NestedTensorCollectionUtils()
        assert utils.all_close(result['tensors'], tensors)
        assert utils.all_close(result['losses'], losses)


if __name__ == '__main__':
    test = TestFGD()
    test.generate_result()
