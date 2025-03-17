# pylint: disable=duplicate-code

__all__ = [
    'CLIPViT',
    'CLIPText',
]

from abc import ABC
from typing import cast
from typing_extensions import Self

import einops
import torch
import torch.nn.functional as F
from torch import nn

from ...patches.py_ import remove_prefix
from ...patches.torch import Sequential
from ...utils import StateDict, StateDictConverter, set_temp
from ...utils.state_dicts import parallel_conversion
from .pretrained import PretrainedMixin
from .transformer import Block, Transformer
from .utils import ApproximateGELU
from .vit import ViT


class CLIPBlocksStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'ln_(1|2)\.(.*)', r'_norm\1.\2')
        self._register_regex_converter(r'attn\.(.*)', r'_attention.\1')
        self._register_regex_converter(r'mlp\.c_fc\.(.*)', r'_mlp.0.\1')
        self._register_regex_converter(r'mlp\.c_proj\.(.*)', r'_mlp.2.\1')

    def convert(self, state_dict: StateDict) -> StateDict:
        state_dict = {
            remove_prefix(k, 'resblocks.'): v
            for k, v in state_dict.items()
        }
        super_ = super()

        @parallel_conversion
        def func(self: Self, state_dict: StateDict, prefix: str) -> StateDict:
            module = cast(Sequential, self._module)
            with set_temp(self, '._module', module[int(prefix)]):
                return super_.convert(state_dict)  # type: ignore[attr-defined]

        return func(self, state_dict)  # pylint: disable=no-value-for-parameter


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

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        state_dict = super()._pre_convert(state_dict)
        state_dict = {
            k.removeprefix('visual.'): v
            for k, v in state_dict.items()
            if k.startswith('visual.')
        }
        return state_dict


class CLIPViTStateDictConverter(CLIPVisionStateDictConverterMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'conv1\.(.*)', r'_patch_embedding.\1')
        self._register_key_mapping('class_embedding', '_cls_token')
        self._register_key_mapping(
            'positional_embedding',
            '_position_embedding',
        )
        self._register_regex_converter(r'ln_pre\.(.*)', r'_pre_norm.\1')
        self._register_child_converter(
            'transformer',
            '_blocks',
            CLIPBlocksStateDictConverter,
        )
        self._register_regex_converter(r'ln_post\.(.*)', r'_norm.\1')
        self._register_key_mapping('proj', '_projector')


class CLIPBlock(Block):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._norm1.eps = 1e-5
        self._norm2.eps = 1e-5
        self._mlp[1].__class__ = ApproximateGELU


class CLIPViT(CLIPMixin, ViT):
    BLOCK_TYPE = CLIPBlock
    STATE_DICT_CONVERTER = CLIPViTStateDictConverter

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._patch_embedding.bias = None
        self._pre_norm = nn.LayerNorm(self.width)

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
            mode='bilinear',
        )

        x = x + position_embedding
        x = self._pre_norm(x)
        x = self._blocks(x)
        x = self._norm(x)

        x = self._project(x)

        if normalize:
            x = F.normalize(x, dim=-1)

        cls_ = x[:, 0]
        x = x[:, 1:]

        if return_2d:
            x = einops.rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return cls_, x


class CLIPTextStateDictConverter(CLIPStateDictConverterMixin):

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

        self._register_regex_converter(r'visual\..*', None)
        self._register_key_mapping('input_resolution', None)
        self._register_key_mapping('context_length', None)
        self._register_key_mapping('vocab_size', None)
        self._register_key_mapping('logit_scale', None)


class CLIPText(CLIPMixin, Transformer):
    BLOCK_TYPE = CLIPBlock
    STATE_DICT_CONVERTER = CLIPTextStateDictConverter

    @classmethod
    def eos(cls, text: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b = text.shape[0]
        eos_indices = text.argmax(-1)  # <EOS> has the highest number
        return x[torch.arange(b), eos_indices]

    def forward(
        self,
        text: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        x = super().forward(text)
        x = self._project(x)

        if normalize:
            x = F.normalize(x, dim=-1)

        return x
