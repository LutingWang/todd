# pylint: disable=duplicate-code

__all__ = [
    'EVA_CLIPViT',
    'EVA_CLIPText',
]

import enum
from typing import cast

import einops
import torch
import torch.nn.functional as F
from torch import nn

from ...patches.torch import Sequential
from ...utils import StateDict, StateDictConverter
from ...utils.state_dicts import SequentialStateDictConverterMixin
from .attentions import BaseAttention
from .clip import (
    CLIPBlock,
    CLIPBlocksStateDictConverter,
    CLIPText,
    CLIPTextStateDictConverter,
)
from .dino import DINOStateDictConverter
from .position_embeddings import (
    rotary_position_embedding_2d,
    sinusoidal_position_embedding,
)
from .pretrained import PretrainedMixin
from .transformer import mlp
from .utils import SwiGLU
from .vit import ViT


class ViTAttentionStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_key_mapping('qkv.weight', '_in_proj.weight')
        self._register_regex_converter(r'(q|v)_bias', r'_\g<0>')
        self._register_regex_converter(r'inner_attn_ln\.(.*)', r'_norm.\1')
        self._register_regex_converter(r'proj\.(.*)', r'_out_proj.\1')

        self._register_regex_converter(r'rope\.freqs_(sin|cos)', None)

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        state_dict = super()._pre_convert(state_dict)

        module = cast(ViTAttention, self._module)
        if module.with_norm:
            state_dict['qkv.weight'] = torch.cat((
                state_dict.pop('q_proj.weight'),
                state_dict.pop('k_proj.weight'),
                state_dict.pop('v_proj.weight'),
            ))

        return state_dict


class ViTAttention(PretrainedMixin, BaseAttention):
    STATE_DICT_CONVERTER = ViTAttentionStateDictConverter

    def __init__(
        self,
        *args,
        with_norm: bool,
        rope: bool,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._with_norm = with_norm
        self._rope = rope

        self._in_proj = nn.Linear(self._width, self.hidden_dim * 3, False)
        self._q_bias = nn.Parameter(torch.empty(self.hidden_dim))
        self._v_bias = nn.Parameter(torch.empty(self.hidden_dim))
        self._norm = (
            nn.LayerNorm(self.hidden_dim, 1e-6) if with_norm else nn.Identity()
        )
        self._out_proj = nn.Linear(self.hidden_dim, self._width)

    @property
    def with_norm(self) -> bool:
        return self._with_norm

    def _apply_rope(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        i = rope.shape[0]**2
        x1 = x[..., :-i, :]
        x2 = x[..., -i:, :]
        x2 = rotary_position_embedding_2d(x2, rope)
        x = torch.cat((x1, x2), -2)
        return x

    def forward(
        self,
        x: torch.Tensor,
        wh: tuple[int, int],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv = self._in_proj(x)
        qkv = einops.rearrange(qkv, 'b n (qkv c) -> qkv b n c', qkv=3)
        q, k, v = torch.unbind(qkv)
        q = q + self._q_bias
        v = v + self._v_bias
        q = einops.rearrange(q, 'b n (nh c) -> b nh n c', nh=self._num_heads)
        k = einops.rearrange(k, 'b n (nh c) -> b nh n c', nh=self._num_heads)
        v = einops.rearrange(v, 'b n (nh c) -> b nh n c', nh=self._num_heads)

        if self._rope:
            num_patches, = set(wh)
            rope = sinusoidal_position_embedding(
                torch.linspace(0, 16, num_patches + 1, device=x.device)[:-1],
                self.head_dim // 2 + 2,
            )[..., :-2]
            q = self._apply_rope(q, rope)
            k = self._apply_rope(k, rope)

        x = F.scaled_dot_product_attention(q, k, v, attention_mask)
        x = einops.rearrange(x, 'b nh n c -> b n (nh c)')

        x = self._norm(x)
        x = self._out_proj(x)
        return x


class SwiGLUStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'w1\.(.*)', r'_linear1.\1')
        self._register_regex_converter(r'w2\.(.*)', r'_linear2.\1')
        self._register_regex_converter(r'ffn_ln\.(.*)', r'_norm.\1')
        self._register_regex_converter(r'w3\.(.*)', r'_projector.\1')


class MLPStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'fc1\.(.*)', r'0.\1')
        self._register_regex_converter(r'fc2\.(.*)', r'2.\1')


class MLPEnum(enum.Enum):
    SWIGLU = 'swiglu'
    MLP = 'mlp'


class ViTBlocksStateDictConverter(
    SequentialStateDictConverterMixin,
    StateDictConverter,
):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_child_converter(
            'attn',
            '_attention',
            ViTAttentionStateDictConverter,
        )
        self._register_regex_converter(r'norm(1|2)\..*', r'_\g<0>')

        mlp_type, = set(
            cast(ViTBlock, block).mlp_type
            for block in cast(Sequential, self._module)
        )
        mlp_state_dict_converter: type[StateDictConverter]
        match mlp_type:
            case MLPEnum.SWIGLU:
                mlp_state_dict_converter = SwiGLUStateDictConverter
            case MLPEnum.MLP:
                mlp_state_dict_converter = MLPStateDictConverter
            case _:
                raise ValueError(f"Unknown mlp type: {mlp_type}")
        self._register_child_converter('mlp', '_mlp', mlp_state_dict_converter)

        self._register_regex_converter(r'gamma_(1|2)', r'_scaler\1')

    def _post_convert(self, state_dict: StateDict) -> StateDict:
        module = cast(ViTBlock, self._module)
        state_dict.setdefault('_scaler1', torch.ones_like(module._scaler1))
        state_dict.setdefault('_scaler2', torch.ones_like(module._scaler2))
        return super()._post_convert(state_dict)


class ViTBlock(nn.Module):

    def __init__(
        self,
        *args,
        width: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        post_norm: bool = False,
        attention_with_norm: bool = False,
        rope: bool = False,
        mlp_type: MLPEnum | str = MLPEnum.MLP,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._post_norm = post_norm

        self._norm1 = nn.LayerNorm(width, 1e-6)
        self._attention = ViTAttention(
            width=width,
            num_heads=num_heads,
            with_norm=attention_with_norm,
            rope=rope,
        )
        self._norm2 = nn.LayerNorm(width, 1e-6)

        if not isinstance(mlp_type, MLPEnum):
            mlp_type = MLPEnum(mlp_type)
        self._mlp_type = mlp_type

        mlp_: nn.Module
        match mlp_type:
            case MLPEnum.SWIGLU:
                mlp_ = SwiGLU(
                    in_features=width,
                    hidden_features=int(width * mlp_ratio),
                    out_features=width,
                )
            case MLPEnum.MLP:
                mlp_ = mlp(width, int(width * mlp_ratio), width)
            case _:
                raise ValueError(f"Unknown mlp type: {mlp_type}")
        self._mlp = mlp_

        self._scaler1 = nn.Parameter(torch.empty(width))
        self._scaler2 = nn.Parameter(torch.empty(width))

    @property
    def mlp_type(self) -> MLPEnum:
        return self._mlp_type

    def forward(self, x: torch.Tensor, *, wh: tuple[int, int]) -> torch.Tensor:
        if self._post_norm:
            x = x + self._scaler1 * self._norm1(self._attention(x, wh))
            x = x + self._scaler2 * self._norm2(self._mlp(x))
        else:
            x = x + self._scaler1 * self._attention(self._norm1(x), wh)
            x = x + self._scaler2 * self._mlp(self._norm2(x))
        return x


class ViTStateDictConverter(DINOStateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_child_converter(
            'blocks',
            '_blocks',
            ViTBlocksStateDictConverter,
        )
        self._register_regex_converter(r'head\.(.*)', r'_projector.\1')

        self._register_regex_converter(r'.*freqs_(sin|cos).*', None)

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        state_dict = super()._pre_convert(state_dict)
        state_dict = {
            k.removeprefix('visual.'): v
            for k, v in state_dict.items()
            if k.startswith('visual.')
        }
        return state_dict


class EVA_CLIPViT(PretrainedMixin, ViT):  # noqa: E501 N801 pylint: disable=invalid-name
    BLOCK_TYPE = ViTBlock
    STATE_DICT_CONVERTER = ViTStateDictConverter

    def __init__(self, *args, out_features: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._projector = nn.Linear(self.width, out_features)

    def forward(
        self,
        image: torch.Tensor,
        return_2d: bool,
        normalize: bool = True,
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
        x = self._blocks(x, wh=(w, h))
        x = self._norm(x)
        x = self._projector(x)

        if normalize:
            x = F.normalize(x, dim=-1)

        cls_ = x[:, 0]
        x = x[:, 1:]

        if return_2d:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return cls_, x


class TextBlock(CLIPBlock):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mlp[1].__class__ = nn.GELU


class TextStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'token_embedding\..*', r'_\g<0>')
        self._register_key_mapping(
            'positional_embedding',
            '_position_embedding',
        )
        self._register_child_converter(
            'transformer',
            '_blocks',
            CLIPBlocksStateDictConverter,
        )
        self._register_regex_converter(r'ln_final\.(.*)', r'_norm.\1')
        self._register_key_mapping(r'text_projection', r'_projector')

        self._register_key_mapping('logit_scale', None)

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        state_dict = super()._pre_convert(state_dict)
        state_dict = {
            k.removeprefix('text.'): v
            for k, v in state_dict.items()
            if k.startswith('text.')
        }
        return state_dict


class EVA_CLIPText(CLIPText):  # noqa: E501 N801 pylint: disable=invalid-name
    BLOCK_TYPE = TextBlock
    STATE_DICT_CONVERTER = cast(
        type[CLIPTextStateDictConverter],
        TextStateDictConverter,
    )
