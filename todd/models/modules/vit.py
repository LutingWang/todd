__all__ = [
    'ViT',
]

from types import MappingProxyType
from typing import Any, Mapping

import einops
import torch
from torch import nn

from ...patches.torch import Sequential
from ..utils import interpolate_position_embedding
from .transformer import Block


class ViT(nn.Module):
    BLOCK_TYPE: type[nn.Module] = Block

    def __init__(
        self,
        *args,
        in_channels: int = 3,
        patch_size: int = 16,
        patch_wh: tuple[int, int] = (14, 14),
        width: int = 768,
        depth: int = 12,
        block_kwargs: Mapping[str, Any] = MappingProxyType(dict()),  # noqa: B006 E501 pylint: disable=line-too-long
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._patch_size = patch_size
        self._patch_wh = patch_wh

        self._patch_embedding = nn.Conv2d(
            in_channels,
            width,
            patch_size,
            patch_size,
        )
        self._cls_token = nn.Parameter(torch.empty(width))
        self._position_embedding = nn.Parameter(
            torch.empty(self.num_patches + 1, width),
        )

        self._blocks = Sequential(
            *[
                self.BLOCK_TYPE(width=width, **block_kwargs)
                for _ in range(depth)
            ],
        )

        self._norm = nn.LayerNorm(width, 1e-6)

    @property
    def in_channels(self) -> int:
        return self._patch_embedding.in_channels

    @property
    def num_patches(self) -> int:
        w, h = self._patch_wh
        return w * h

    @property
    def width(self) -> int:
        return self._cls_token.numel()

    @property
    def depth(self) -> int:
        return len(self._blocks)

    def _interpolate_position_embedding(
        self,
        wh: tuple[int, int],
        **kwargs,
    ) -> torch.Tensor:
        position_embedding = interpolate_position_embedding(
            self._position_embedding,
            self._patch_wh,
            wh,
            **kwargs,
        )
        position_embedding = einops.rearrange(
            position_embedding,
            'n c -> 1 n c',
        )
        return position_embedding

    def forward(
        self,
        image: torch.Tensor,
        return_2d: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self._patch_embedding(image)

        b, _, h, w = x.shape
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        cls_token = einops.repeat(self._cls_token, 'd -> b 1 d', b=b)
        x = torch.cat((cls_token, x), 1)

        position_embedding = self._interpolate_position_embedding(
            (w, h),
            mode='bicubic',
        )

        x = x + position_embedding
        x = self._blocks(x)
        x = self._norm(x)

        cls_ = x[:, 0]
        x = x[:, 1:]

        if return_2d:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return cls_, x
