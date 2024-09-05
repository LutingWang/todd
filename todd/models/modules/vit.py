__all__ = [
    'ViT',
]

import einops
import torch
from torch import nn

from ...bases.configs import Config
from ...patches.torch import Sequential
from ...registries import InitWeightsMixin
from ..utils import interpolate_position_embedding
from .transformer import Block


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


class ViT(InitWeightsMixin, nn.Module):
    BLOCK_TYPE = Block

    def __init__(
        self,
        *args,
        patch_size: int = 16,
        patch_wh: tuple[int, int] = (14, 14),
        width: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._patch_size = patch_size
        self._patch_wh = patch_wh
        self._num_heads = num_heads

        self._patch_embedding = nn.Conv2d(3, width, patch_size, patch_size)
        self._cls_token = nn.Parameter(torch.empty(width))
        self._position_embedding = nn.Parameter(
            torch.empty(self.num_patches + 1, width),
        )
        self._blocks = Sequential(
            *[
                self.BLOCK_TYPE(width=width, num_heads=num_heads)
                for _ in range(depth)
            ],
        )
        self._norm = nn.LayerNorm(width, 1e-6)

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

    def init_weights(self, config: Config) -> bool:
        super().init_weights(config)
        nn.init.trunc_normal_(self._position_embedding, std=.02)
        nn.init.trunc_normal_(self._cls_token, std=.02)
        self.apply(init_weights)
        return False

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
            '... -> 1 ...',
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
        x = torch.cat((cls_token, x), dim=1)

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
