__all__ = [
    'DINO',
]

import einops
import torch
from torch import nn

from ..bases.configs import Config
from ..patches.torch import load_state_dict
from ..registries import InitWeightsMixin
from ..utils import StateDict, StateDictConverter
from .utils import interpolate_position_embedding


class DINOStateDictConverter(StateDictConverter):

    def _convert_patch_embed(self, key: str) -> str:
        assert key.startswith('proj.')
        key = key.removeprefix('proj.')
        return f'_patch_embedding.{key}'

    def _convert_block_attn(self, key: str) -> str:
        if key.startswith('qkv.'):
            key = key.removeprefix('qkv.')
            key = f'in_proj_{key}'
        elif key.startswith('proj.'):
            key = f'out_{key}'
        return f'_attention.{key}'

    def _convert_block_mlp(self, key: str) -> str:
        if key.startswith('fc1.'):
            key = key.removeprefix('fc1.')
            key = '0.' + key
        elif key.startswith('act.'):
            key = key.removeprefix('act.')
            key = '1.' + key
        elif key.startswith('fc2.'):
            key = key.removeprefix('fc2.')
            key = '2.' + key
        else:
            raise ValueError(f"Unknown key: {key}")
        return f'_mlp.{key}'

    def _convert_block(self, key: str) -> str:
        if key.startswith(('norm1.', 'norm2.')):
            return f'_{key}'
        if key.startswith('attn.'):
            key = key.removeprefix('attn.')
            return self._convert_block_attn(key)
        if key.startswith('mlp.'):
            key = key.removeprefix('mlp.')
            return self._convert_block_mlp(key)
        return key

    def _convert_blocks(self, key: str) -> str:
        index = key.index('.') + 1
        prefix = key[:index]
        key = key[index:]
        return '_blocks.' + prefix + self._convert_block(key)

    def _convert(self, key: str) -> str | None:
        if key.startswith('patch_embed.'):
            key = key.removeprefix('patch_embed.')
            return self._convert_patch_embed(key)
        if key.startswith('blocks'):
            key = key.removeprefix('blocks.')
            return self._convert_blocks(key)
        if key.startswith('norm.'):
            return f'_{key}'
        if key == 'cls_token':
            return f'_{key}'
        if key == 'pos_embed':
            return '_position_embedding'
        return key

    def _post_convert(self, state_dict: StateDict) -> StateDict:
        state_dict['_cls_token'] = einops.rearrange(
            state_dict['_cls_token'],
            '1 1 d -> d',
        )
        state_dict['_position_embedding'] = einops.rearrange(
            state_dict['_position_embedding'],
            '1 l d -> l d',
        )
        return state_dict


def mlp(
    in_features: int,
    hidden_features: int,
    out_features: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.GELU(),
        nn.Linear(hidden_features, out_features),
    )


class Block(nn.Module):

    def __init__(
        self,
        *args,
        width: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._norm1 = nn.LayerNorm(width, eps=1e-6)
        self._attention = nn.MultiheadAttention(
            width,
            num_heads,
            batch_first=True,
        )
        self._norm2 = nn.LayerNorm(width, eps=1e-6)
        self._mlp = mlp(width, int(width * mlp_ratio), width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self._norm1(x)
        attention, _ = self._attention(norm, norm, norm, need_weights=False)
        x = x + attention
        norm = self._norm2(x)
        mlp_ = self._mlp(norm)
        x = x + mlp_
        return x


class DINO(InitWeightsMixin, nn.Module):

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
        self._width = width

        self._patch_embedding = nn.Conv2d(3, width, patch_size, patch_size)
        self._cls_token = nn.Parameter(torch.empty(width))
        self._position_embedding = nn.Parameter(
            torch.empty(self.num_patches + 1, width),
        )
        self._blocks = nn.Sequential(
            *[Block(width=width, num_heads=num_heads) for _ in range(depth)],
        )
        self._norm = nn.LayerNorm(width, eps=1e-6)

    @property
    def num_patches(self) -> int:
        w, h = self._patch_wh
        return w * h

    @property
    def width(self) -> int:
        return self._width

    def init_weights(self, config: Config) -> bool:
        super().init_weights(config)
        nn.init.trunc_normal_(self._position_embedding, std=.02)
        nn.init.trunc_normal_(self._cls_token, std=.02)

        def f(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        self.apply(f)
        return False

    def load_pretrained(self, *args, **kwargs) -> None:
        converter = DINOStateDictConverter()
        state_dict = converter.load(*args, **kwargs)
        state_dict = converter.convert(state_dict)
        load_state_dict(self, state_dict, strict=False)

    def forward(
        self,
        image: torch.Tensor,
        return_2d: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self._patch_embedding(image)

        b, _, h, w = x.shape
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = einops.repeat(self._cls_token, 'd -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        position_embedding = interpolate_position_embedding(
            self._position_embedding,
            self._patch_wh,
            (w, h),
            mode='bicubic',
        )
        position_embedding = einops.rearrange(
            position_embedding,
            '... -> 1 ...',
        )

        x = x + position_embedding
        x = self._blocks(x)
        x = self._norm(x)

        cls_ = x[:, 0]
        x = x[:, 1:]

        if return_2d:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return cls_, x
