from datetime import datetime
import os
from typing import List, Literal, Optional

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F

from ..utils import get_iter
from .base import BaseVisual
from .builder import VISUALS


@VISUALS.register_module()
class ActivationVisual(BaseVisual):
    def __init__(self, log_dir: str, log_name: Optional[str] = None, mode: Literal['image', 'tensor', 'overlay'] = 'overlay'):
        super().__init__()
        if log_name is None:
            log_name = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        log_dir = os.path.join(log_dir, log_name)
        os.makedirs(log_dir)

        if mode != 'overlay':
            raise NotImplementedError

        self._log_dir = log_dir
        self._mode = mode

    def forward(self, images: List[np.ndarray], tensors: torch.Tensor, **kwargs):
        """Activation visualizer.

        Args:
            images: :math:`(N, H, W, 3)`
            tensors: :math:`(N, C, H, W)`
        """
        tensors = einops.reduce(tensors, 'n c h w -> n 1 h w', reduction='mean')
        kwargs = '_'.join(k + str(v) for k, v in kwargs.items())
        for i, image in enumerate(images):
            image_path = os.path.join(
                self._log_dir, f'iter{get_iter()}_image{i}_{kwargs}_{self._mode}.png',
            )
            h, w, c = image.shape
            tensor: torch.Tensor = F.interpolate(tensors[[i]], (h, w), mode='bilinear')[0, 0]
            tensor = (tensor.max() - tensor) / (tensor.max() - tensor.min()) * 255
            tensor = tensor.detach().cpu().numpy().astype(np.uint8)
            tensor = cv2.applyColorMap(tensor, cv2.COLORMAP_JET)
            if self._mode == 'overlay':
                overlay = image * 0.5 + tensor * 0.5
                cv2.imwrite(image_path, overlay)
