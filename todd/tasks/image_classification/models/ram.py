__all__ = [
    'RAMplus',
]

import pathlib
from dataclasses import dataclass
from typing import Any, Never, cast
from typing_extensions import Self

import einops
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import SwinTransformer
from torchvision.models.swin_transformer import ShiftedWindowAttention
from transformers import BertConfig, BertModel
from transformers.models.bert import BertLayer
from transformers.models.bert.modeling_bert import BertSelfAttention

from todd.models.modules import PretrainedMixin
from todd.utils import StateDict, StateDictConverter
from todd.utils.state_dicts import SequentialStateDictConverterMixin


class RAMStateDictConverterMixin(StateDictConverter):

    def load(self, *args, **kwargs) -> StateDict:
        checkpoint = super().load(*args, **kwargs)
        return cast(StateDict, checkpoint['model'])


@dataclass(frozen=True)
class Category:
    name: str
    chinese_name: str
    threshold: float


class Categories(pd.DataFrame):

    @classmethod
    def load(cls, f: Any = None) -> Self:
        if f is None:
            f = pathlib.Path(__file__).with_suffix('.csv')
        df = pd.read_csv(
            f,
            header=None,
            names=list(Category.__dataclass_fields__),  # noqa: E501 pylint: disable=no-member
        )
        df.__class__ = cls
        return cast(Self, df)

    def decode(self, logits: torch.Tensor) -> list[list[str]]:
        prob = logits.sigmoid()
        thresholds = torch.from_numpy(self.threshold.to_numpy()).to(logits)
        preds = prob > thresholds

        i: int
        j: int
        outputs: list[list[str]] = [[] for _ in range(logits.shape[0])]
        for i, j in preds.nonzero().tolist():
            outputs[i].append(self.name[j])

        return outputs


class SwinTransformerBlockStateDictConverter(
    SequentialStateDictConverterMixin,
    StateDictConverter,
):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'mlp\.fc1\.(.*)', r'mlp.0.\1')
        self._register_regex_converter(r'mlp\.fc2\.(.*)', r'mlp.3.\1')

        self._register_key_mapping('attn.relative_position_index', None)
        self._register_key_mapping('attn_mask', None)

    def _convert(self, key: str) -> str | None:
        if key in (
            'norm1.weight',
            'norm1.bias',
            'attn.qkv.weight',
            'attn.qkv.bias',
            'attn.relative_position_bias_table',
            'attn.proj.weight',
            'attn.proj.bias',
            'norm2.weight',
            'norm2.bias',
        ):
            return key
        return super()._convert(key)


class EncoderStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(
            r'patch_embed\.proj\.(.*)',
            r'features.0.0.\1',
        )
        self._register_regex_converter(
            r'patch_embed\.norm\.(.*)',
            r'features.0.2.\1',
        )

    def _pre_convert_layer(
        self,
        state_dict: StateDict,
        i: int,
        layer: nn.Sequential,
    ) -> StateDict:
        layer_state_dict: StateDict = dict()

        prefix = f'layers.{i // 2}.'
        prefix += 'blocks.' if i % 2 == 0 else 'downsample.'

        for k in list(state_dict):
            if k.startswith(prefix):
                layer_state_dict[k.removeprefix(prefix)] = state_dict.pop(k)

        if i % 2 == 0:
            converter = SwinTransformerBlockStateDictConverter(module=layer)
            layer_state_dict = converter.convert(layer_state_dict)  # noqa: E501 pylint: disable=no-value-for-parameter

        return {
            f'features.{i + 1}.{k}': v
            for k, v in layer_state_dict.items()
        }

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        state_dict = super()._pre_convert(state_dict)

        layer_state_dict: StateDict = dict()

        module = cast(Encoder, self._module)
        for i, layer in enumerate(module.features[1:]):
            # `_pre_convert_layer` removes keys from `state_dict`
            layer_state_dict |= self._pre_convert_layer(state_dict, i, layer)

        return layer_state_dict | state_dict

    def _convert(self, key: str) -> str | None:
        if key.startswith('features.'):
            return key
        if key in ('norm.weight', 'norm.bias'):
            return key
        return super()._convert(key)


class Encoder(PretrainedMixin, SwinTransformer):
    STATE_DICT_CONVERTER = EncoderStateDictConverter

    def __init__(
        self,
        patch_size: int,
        window_size: int,
        width: int,
        depths: tuple[int, ...],
        num_heads: tuple[int, ...],
        hidden_channels: int,
    ) -> None:
        super().__init__(
            (patch_size, patch_size),
            width,
            depths,
            num_heads,
            (window_size, window_size),
            num_classes=hidden_channels,
        )
        for module in self.modules():
            if isinstance(module, ShiftedWindowAttention):
                module._non_persistent_buffers_set.add(
                    'relative_position_index',
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        cls_ = self.avgpool(x)
        cls_ = self.flatten(cls_)
        x = torch.cat(
            [
                einops.rearrange(cls_, 'b c -> b 1 c'),
                einops.rearrange(x, 'b c h w -> b (h w) c'),
            ],
            1,
        )
        x = self.head(x)
        return x


class BertEmbeddings(nn.Module):

    def forward(
        self,
        *args,
        inputs_embeds: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return inputs_embeds


class BertAttention(nn.Module):

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[Never, ...]]:
        return hidden_states, tuple()


class Decoder(BertModel):  # pylint: disable=abstract-method

    def __init__(
        self,
        hidden_channels: int,
        depth: int,
        num_heads: int,
    ) -> None:
        config = BertConfig(
            num_attention_heads=num_heads,
            num_hidden_layers=depth,
            add_cross_attention=True,
            is_decoder=True,
            return_dict=False,
        )
        super().__init__(config, False)

        self.embeddings = BertEmbeddings()

        layer: BertLayer
        for layer in self.encoder.layer:
            layer.attention = BertAttention()
            self_attention: BertSelfAttention = layer.crossattention.self
            out_channels = self_attention.all_head_size
            self_attention.key = nn.Linear(hidden_channels, out_channels)
            self_attention.value = nn.Linear(hidden_channels, out_channels)

        self._in_linear = nn.Linear(hidden_channels, config.hidden_size)
        self._out_linear = nn.Linear(config.hidden_size, 1)

    def forward(  # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
        category_embedding: torch.Tensor,
    ) -> torch.Tensor:
        category_embedding = self._in_linear(category_embedding)
        category_embedding = category_embedding.relu()

        embedding, *_ = super().forward(  # pylint: disable=no-member
            inputs_embeds=category_embedding,
            encoder_hidden_states=x,
        )

        logits = self._out_linear(embedding)
        logits = einops.rearrange(logits, 'b k 1 -> b k')
        return logits


class RAMplusStateDictConverterMixin(RAMStateDictConverterMixin):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_child_converter(
            'visual_encoder',
            '_encoder',
            EncoderStateDictConverter,
        )
        self._register_regex_converter(
            r'image_proj\.(.*)',
            r'_encoder.head.\1',
        )
        self._register_key_mapping('label_embed', '_category_embedding')
        self._register_regex_converter(
            r'wordvec_proj\.(.*)',
            r'_decoder._in_linear.\1',
        )
        self._register_regex_converter(r'tagging_head\.(.*)', r'_decoder.\1')
        self._register_regex_converter(r'fc\.(.*)', r'_decoder._out_linear.\1')

        self._register_key_mapping('logit_scale', None)

    def _post_convert(self, state_dict: StateDict) -> StateDict:
        module = cast(RAMplus, self._module)

        state_dict['_category_embedding'] = einops.rearrange(
            state_dict['_category_embedding'],
            '(k n) c -> k n c',
            n=module.num_descriptions,
        )

        return super()._post_convert(state_dict)


class RAMplus(PretrainedMixin, nn.Module):
    STATE_DICT_CONVERTER = RAMplusStateDictConverterMixin

    def __init__(
        self,
        *args,
        patch_size: int = 4,
        window_size: int = 12,
        encoder_width: int = 192,
        encoder_depths: tuple[int, ...] = (2, 2, 18, 2),
        encoder_num_heads: tuple[int, ...] = (6, 12, 24, 48),
        hidden_channels: int = 512,
        num_categories: int,
        num_descriptions: int = 51,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._encoder = Encoder(
            patch_size,
            window_size,
            encoder_width,
            encoder_depths,
            encoder_num_heads,
            hidden_channels,
        )
        self._category_embedding = nn.Parameter(
            torch.empty(
                num_categories,
                num_descriptions,
                hidden_channels,
            ),
        )
        self._decoder = Decoder(
            hidden_channels,
            decoder_depth,
            decoder_num_heads,
        )

    @property
    def num_descriptions(self) -> int:
        _, num_descriptions, _ = self._category_embedding.shape
        return num_descriptions

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self._encoder(image)

        cls_ = x[:, 0]
        weights = torch.einsum(
            'b c, k n c -> b k n',
            F.normalize(cls_, dim=-1),
            self._category_embedding,
        )
        weights = weights / 0.07
        weights = weights.softmax(2)
        category_embedding = torch.einsum(
            'b k n, k n c -> b k c',
            weights,
            self._category_embedding,
        )

        logits = self._decoder(x, category_embedding)
        return logits
