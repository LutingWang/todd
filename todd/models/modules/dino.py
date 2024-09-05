__all__ = [
    'DINOStateDictConverter',
    'DINO',
    'DINOv2StateDictConverter',
    'DINOv2',
]

import einops
import torch
from torch import nn

from ...utils import StateDict, StateDictConverter
from .pretrained import PretrainedMixin
from .transformer import Block
from .vit import ViT


class DINOStateDictConverterMixin(StateDictConverter):

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


class DINOMixin(PretrainedMixin, ViT):

    def _interpolate_position_embedding(
        self,
        wh: tuple[int, int],
        **kwargs,
    ) -> torch.Tensor:
        kwargs.setdefault('offset', 0.1)
        return super()._interpolate_position_embedding(wh, **kwargs)


class DINOStateDictConverter(DINOStateDictConverterMixin):
    pass


class DINO(DINOMixin):
    STATE_DICT_CONVERTER = DINOStateDictConverter


class DINOv2Scaler(nn.Module):

    def __init__(self, *args, width: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scaler = nn.Parameter(torch.empty(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._scaler


class DINOv2Block(Block):

    def __init__(self, *args, width: int, **kwargs) -> None:
        super().__init__(*args, width=width, **kwargs)
        self._scaler1 = DINOv2Scaler(width=width)
        self._scaler2 = DINOv2Scaler(width=width)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm = self._norm1(x)
        attention, _ = self._attention(
            norm,
            norm,
            norm,
            need_weights=False,
            attn_mask=attention_mask,
        )
        x = x + self._scaler1(attention)
        norm = self._norm2(x)
        mlp_ = self._mlp(norm)
        x = x + self._scaler2(mlp_)
        return x


class DINOv2StateDictConverter(DINOStateDictConverterMixin):

    def _convert_block(self, key: str) -> str:
        if key.startswith(('ls1.', 'ls2.')):
            prefix, key = key.split('.', 1)
            assert prefix.startswith('ls')
            assert key == 'gamma'
            prefix = prefix.removeprefix('ls')
            return f'_scaler{prefix}._scaler'
        return super()._convert_block(key)

    def _post_convert(self, state_dict: StateDict) -> StateDict:
        state_dict.pop('mask_token')
        return super()._post_convert(state_dict)


class DINOv2(DINOMixin):
    BLOCK_TYPE = DINOv2Block
    STATE_DICT_CONVERTER = DINOv2StateDictConverter
