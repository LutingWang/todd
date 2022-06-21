from typing import List, Literal

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F

from .base import VISUALS, BaseVisual

__all__ = [
    'ActivationVisual',
]


@VISUALS.register_module()
class ActivationVisual(BaseVisual):

    def __init__(
        self,
        *args,
        mode: Literal['image', 'tensor', 'overlay'] = 'overlay',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert mode == 'overlay', 'Only support overlay mode.'
        self._mode = mode

    def forward(
        self,
        images: List[np.ndarray],
        tensors: torch.Tensor,
    ) -> List[np.ndarray]:
        """Activation visualizer.

        Args:
            images: :math:`(N, H, W, 3)`
            tensors: :math:`(N, C, H, W)`
        """
        tensors = einops.reduce(
            tensors,
            'n c h w -> n 1 h w',
            reduction='mean',
        )
        results = []
        for i, image in enumerate(images):
            h, w, c = image.shape
            tensor: torch.Tensor = F.interpolate(
                tensors[[i]], (h, w), mode='bilinear'
            )[0, 0]
            tensor = (  # yapf: disable
                (tensor.max() - tensor)
                / (tensor.max() - tensor.min())
                * 255
            )
            tensor = tensor.detach().cpu().numpy().astype(np.uint8)
            tensor = cv2.applyColorMap(tensor, cv2.COLORMAP_JET)
            if self._mode == 'overlay':
                result = image * 0.5 + tensor * 0.5
            results.append(result)
        return results
