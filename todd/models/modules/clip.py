# pylint: disable=duplicate-code

__all__ = [
    'CLIPViT',
    'CLIPText',
]

from abc import ABC, abstractmethod

import einops
import torch
import torch.nn.functional as F
from torch import nn

from ...utils import StateDict, StateDictConverter
from .pretrained import PretrainedMixin
from .transformer import Block, Transformer
from .vit import ViT


class CLIPStateDictConverterMixin(StateDictConverter):

    def load(self, *args, **kwargs) -> StateDict:
        f, *args = args  # type: ignore[assignment]
        module: nn.Module = torch.jit.load(f, 'cpu', *args, **kwargs)
        return module.state_dict()


class CLIPMixin(PretrainedMixin, ABC):
    width: int

    def __init__(self, *args, out_features: int | None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if out_features is not None:
            self._projector = nn.Parameter(
                torch.empty(self.width, out_features),
            )

    @property
    def with_projector(self) -> bool:
        return hasattr(self, '_projector')

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_projector:
            return x @ self._projector
        return x


class CLIPVisionStateDictConverterMixin(CLIPStateDictConverterMixin, ABC):

    @abstractmethod
    def _convert_visual(self, key: str) -> str | None:
        pass

    def _convert(self, key: str) -> str | None:
        if key.startswith('visual.'):
            key = key.removeprefix('visual.')
            return self._convert_visual(key)
        return None


class CLIPViTStateDictConverter(CLIPVisionStateDictConverterMixin):

    def _convert_visual_transformer_block_mlp(self, key: str) -> str:
        if key.startswith('c_fc.'):
            key = key.removeprefix('c_fc.')
            key = '0.' + key
        elif key.startswith('gelu.'):
            key = key.removeprefix('gelu.')
            key = '1.' + key
        elif key.startswith('c_proj.'):
            key = key.removeprefix('c_proj.')
            key = '2.' + key
        else:
            raise ValueError(f"Unknown key: {key}")
        return f'_mlp.{key}'

    def _convert_visual_transformer_block(self, key: str) -> str:
        if key.startswith(('ln_1.', 'ln_2.')):
            key = key.removeprefix('ln_')
            return f'_norm{key}'
        if key.startswith('attn.'):
            key = key.removeprefix('attn.')
            return '_attention.' + key
        if key.startswith('mlp.'):
            key = key.removeprefix('mlp.')
            return self._convert_visual_transformer_block_mlp(key)
        return key

    def _convert_visual_transformer(self, key: str) -> str:
        assert key.startswith('resblocks.')
        key = key.removeprefix('resblocks.')
        index = key.index('.') + 1
        prefix = key[:index]
        key = key[index:]
        return (
            '_blocks.' + prefix + self._convert_visual_transformer_block(key)
        )

    def _convert_visual(self, key: str) -> str | None:
        if key.startswith('conv1.'):
            key = key.removeprefix('conv1.')
            return f'_patch_embedding.{key}'
        if key.startswith('transformer.'):
            key = key.removeprefix('transformer.')
            return self._convert_visual_transformer(key)
        if key.startswith('ln_pre.'):
            key = key.removeprefix('ln_pre.')
            return f'_norm_pre.{key}'
        if key.startswith('ln_post.'):
            key = key.removeprefix('ln_post.')
            return f'_norm.{key}'
        if key == 'class_embedding':
            return '_cls_token'
        if key == 'positional_embedding':
            return '_position_embedding'
        if key == 'proj':
            return '_projector'
        return key


class GELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class CLIPBlock(Block):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._norm1.eps = 1e-5
        self._norm2.eps = 1e-5
        self._mlp[1].__class__ = GELU


class CLIPViT(CLIPMixin, ViT):
    BLOCK_TYPE = CLIPBlock
    STATE_DICT_CONVERTER = CLIPViTStateDictConverter

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._patch_embedding.bias = None
        self._norm_pre = nn.LayerNorm(self.width)

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
            mode='bilinear',
        )

        x = x + position_embedding
        x = self._norm_pre(x)
        x = self._blocks(x)
        x = self._norm(x)

        x = self._project(x)

        x = F.normalize(x, dim=-1)

        cls_ = x[:, 0]
        x = x[:, 1:]

        if return_2d:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return cls_, x


class CLIPTextStateDictConverterMixin(CLIPStateDictConverterMixin, ABC):

    def _convert_transformer_block_mlp(self, key: str) -> str:
        if key.startswith('c_fc.'):
            key = key.removeprefix('c_fc.')
            key = '0.' + key
        elif key.startswith('gelu.'):
            key = key.removeprefix('gelu.')
            key = '1.' + key
        elif key.startswith('c_proj.'):
            key = key.removeprefix('c_proj.')
            key = '2.' + key
        else:
            raise ValueError(f"Unknown key: {key}")
        return f'_mlp.{key}'

    def _convert_transformer_block(self, key: str) -> str:
        if key.startswith(('ln_1.', 'ln_2.')):
            key = key.removeprefix('ln_')
            return f'_norm{key}'
        if key.startswith('attn.'):
            key = key.removeprefix('attn.')
            return '_attention.' + key
        if key.startswith('mlp.'):
            key = key.removeprefix('mlp.')
            return self._convert_transformer_block_mlp(key)
        return key

    def _convert_transformer(self, key: str) -> str:
        assert key.startswith('resblocks.')
        key = key.removeprefix('resblocks.')
        index = key.index('.') + 1
        prefix = key[:index]
        key = key[index:]
        return '_blocks.' + prefix + self._convert_transformer_block(key)

    def _convert(self, key: str) -> str | None:
        if key == 'text_projection':
            return '_projector'
        if key.startswith('token_embedding.'):
            return '_' + key
        if key.startswith('transformer.'):
            key = key.removeprefix('transformer.')
            return self._convert_transformer(key)
        if key == 'positional_embedding':
            return '_position_embedding'
        if key.startswith('ln_final.'):
            key = key.removeprefix('ln_final.')
            return f'_norm.{key}'
        if key.startswith('visual.'):
            return None
        if key in (
            'input_resolution',
            'context_length',
            'vocab_size',
            'logit_scale',
        ):
            return None
        return key


class CLIPText(CLIPMixin, Transformer):
    BLOCK_TYPE = CLIPBlock
    STATE_DICT_CONVERTER = CLIPTextStateDictConverterMixin

    @classmethod
    def eos(cls, text: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b = text.shape[0]
        eos_indices = text.argmax(-1)  # <EOS> has the highest number
        return x[torch.arange(b), eos_indices]

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = super().forward(text)
        x = self._project(x)
        return x
