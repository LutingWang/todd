from typing import List

import torch
from mmcv.runner import ModuleList

from .base import ADAPTS, BaseAdapt


@ADAPTS.register_module()
class RoIAlign(BaseAdapt):

    def __init__(self, strides: List[int], *args, **kwargs):
        from mmcv.ops import RoIAlign as RA
        super().__init__(*args, **kwargs)
        self._layers = ModuleList([
            RA(spatial_scale=1 / s, output_size=7, sampling_ratio=0)
            for s in strides
        ])

    def forward(
        self,
        feats: List[torch.Tensor],
        bboxes: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Args:
            feats: l x n x c x h x w
            bboxes: n x r x 4

        Returns:
            roi_feats: l x r x c x 7 x 7
        """
        rois = torch.cat([  # yapf: disable
            torch.cat([b.new_full((b.shape[0], 1), i), b[:, :4]], dim=-1)
            for i, b in enumerate(bboxes)
            if b.shape[0] > 0
        ])
        roi_feats = [layer(f, rois) for layer, f in zip(self._layers, feats)]
        return roi_feats
