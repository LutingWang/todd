__all__ = [
    'draw_activation',
    'reduce',
    'ActivationVisual',
]

from typing import List, Sequence

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F

from .base import VISUALS, BaseVisual


def draw_activation(
    image: np.ndarray,
    activation: torch.Tensor,
    overlay: float = 0.5,
    inverse: bool = False,
) -> np.ndarray:
    """Draw activation map on image.

    Args:
        image: image aligned with the activation
        activation: 2D activation map to be drawn

    Returns:
        activation * overlay + image * (1 - overlay)
    """
    assert 0 <= overlay <= 1
    if overlay == 0:
        return image
    h, w, c = image.shape
    activation = activation.detach()
    activation = einops.rearrange(activation, 'h w -> 1 1 h w')
    activation = F.interpolate(activation, (h, w), mode='bilinear')
    activation = activation[0, 0]
    activation -= activation.min()
    activation /= activation.max()
    if inverse:
        activation = 1 - activation
    activation *= 255
    map_: np.ndarray = activation.cpu().numpy()
    map_ = map_.astype(np.uint8)
    map_ = cv2.applyColorMap(map_, cv2.COLORMAP_JET)
    return map_ * overlay + image * (1 - overlay)


def reduce(
    activation: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Mean the activation.

    Args:
        activation: :math:`(C, H, W)`
        reduction: 'mean'

    Returns:
        :math:`(H, W)`
    """
    if reduction == 'mean':
        return einops.reduce(
            activation,
            'c h w -> h w',
            reduction=reduction,
        )
    raise ValueError(f"Unsupported reduction {reduction}.")


@VISUALS.register_module()
class ActivationVisual(BaseVisual):

    def __init__(
        self,
        *args,
        overlay: float = 0.5,
        reduction: str = 'mean',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._overlay = overlay
        self._reduction = reduction

    def reduce(self, activation: torch.Tensor) -> torch.Tensor:
        return reduce(activation, self._reduction)

    def forward(
        self,
        images: Sequence[np.ndarray],
        activations: Sequence[torch.Tensor],
    ) -> List[np.ndarray]:
        """Activation visualizer.

        Args:
            images: :math:`(N, H, W, 3)`
            activations: :math:`(N, C, H, W)`

        Returns:
            List of :math:`(C, H, W)`
        """
        activations_map = map(self.reduce, activations)
        activations_map = map(draw_activation, images, activations_map)
        return list(activations_map)
