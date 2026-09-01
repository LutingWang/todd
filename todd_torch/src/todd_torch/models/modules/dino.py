# pylint: disable=duplicate-code

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
from ...utils.state_dicts import SequentialStateDictConverterMixin
from .pretrained import PretrainedMixin
from .transformer import Block
from .vit import ViT


class DINOMixin(PretrainedMixin, ViT):

    def _interpolate_position_embedding(
        self,
        wh: tuple[int, int],
        **kwargs,
    ) -> torch.Tensor:
        kwargs.setdefault('offset', 0.1)
        return super()._interpolate_position_embedding(wh, **kwargs)


class DINOBlocksStateDictConverter(
    SequentialStateDictConverterMixin,
    StateDictConverter,
):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'norm[12]\..*', r'_\g<0>')
        self._register_regex_converter(
            r'attn\.qkv\.(.*)',
            r'_attention.in_proj_\1',
        )
        self._register_regex_converter(
            r'attn\.proj\.(.*)',
            r'_attention.out_proj.\1',
        )
        self._register_regex_converter(r'mlp\.fc1\.(.*)', r'_mlp.0.\1')
        self._register_regex_converter(r'mlp\.fc2\.(.*)', r'_mlp.2.\1')


class DINOStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(
            r'patch_embed\.proj\.(.*)',
            r'_patch_embedding.\1',
        )
        self._register_key_mapping('cls_token', '_cls_token')
        self._register_key_mapping('pos_embed', '_position_embedding')
        self._register_child_converter(
            'blocks',
            '_blocks',
            DINOBlocksStateDictConverter,
        )
        self._register_regex_converter(r'norm\..*', r'_\g<0>')

    def _post_convert(self, state_dict: StateDict) -> StateDict:
        state_dict['_cls_token'] = einops.rearrange(
            state_dict['_cls_token'],
            '1 1 d -> d',
        )
        state_dict['_position_embedding'] = einops.rearrange(
            state_dict['_position_embedding'],
            '1 l d -> l d',
        )
        return super()._post_convert(state_dict)


class DINO(DINOMixin):
    STATE_DICT_CONVERTER = DINOStateDictConverter


class DINOv2Scaler(nn.Module):

    def __init__(self, *args, width: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scaler = nn.Parameter(torch.empty(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._scaler


class DINOv2BlocksStateDictConverter(DINOBlocksStateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(
            r'ls(1|2)\.gamma',
            r'_scaler\1._scaler',
        )


class DINOv2Block(PretrainedMixin, Block):
    STATE_DICT_CONVERTER = DINOv2BlocksStateDictConverter

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


class DINOv2StateDictConverter(DINOStateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_child_converter(
            'blocks',
            '_blocks',
            DINOv2BlocksStateDictConverter,
        )

        self._register_key_mapping('mask_token', None)


class DINOv2(DINOMixin):
    BLOCK_TYPE = DINOv2Block
    STATE_DICT_CONVERTER = DINOv2StateDictConverter
