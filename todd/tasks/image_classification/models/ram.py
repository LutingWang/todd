__all__ = [
    'RAMplus',
]

import pathlib
import re
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
from transformers.models.bert import BertConfig, BertLayer, BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention

from todd.models.modules import PretrainedMixin
from todd.utils import StateDict, StateDictConverter


class RAMStateDictConverterMixin(StateDictConverter):

    def load(self, *args, **kwargs) -> StateDict:
        checkpoint = super().load(*args, **kwargs)
        return cast(StateDict, checkpoint['model'])


class RAMplusStateDictConverterMixin(RAMStateDictConverterMixin):

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        state_dict = super()._pre_convert(state_dict)

        pattern = re.compile(
            r'visual_encoder\.layers\.\d+\.blocks\.\d+\.attn\.'
            r'relative_position_index',
        )
        for k in list(filter(pattern.match, state_dict)):
            state_dict.pop(k)

        pattern = re.compile(
            r'visual_encoder\.layers\.\d+\.blocks\.\d+\.attn_mask',
        )
        for k in list(filter(pattern.match, state_dict)):
            state_dict.pop(k)

        state_dict['label_embed'] = einops.rearrange(
            state_dict['label_embed'],
            '(k n) c -> k n c',
            n=RAMplus.NUM_DESCRIPTIONS,
        )

        return state_dict

    def _convert_visual_encoder_patch_embed(self, key: str) -> str:
        if key.startswith('proj.'):
            key = key.removeprefix('proj.')
            return 'features.0.0.' + key
        if key.startswith('norm.'):
            key = key.removeprefix('norm.')
            return 'features.0.2.' + key
        raise ValueError(f"Unknown key: {key}")

    def _convert_visual_encoder_layer_block_mlp(self, key: str) -> str:
        if key.startswith('fc1.'):
            key = key.removeprefix('fc1.')
            key = '0.' + key
        elif key.startswith('fc2.'):
            key = key.removeprefix('fc2.')
            key = '3.' + key
        else:
            raise ValueError(f"Unknown key: {key}")
        return f'mlp.{key}'

    def _convert_visual_encoder_layer_block(self, key: str) -> str:
        if key.startswith('mlp.'):
            key = key.removeprefix('mlp.')
            return self._convert_visual_encoder_layer_block_mlp(key)
        return key

    def _convert_visual_encoder_layer_blocks(self, key: str) -> str:
        index = key.index('.') + 1
        prefix = key[:index]
        key = key[index:]
        return prefix + self._convert_visual_encoder_layer_block(key)

    def _convert_visual_encoder_layer(self, key: str, i: int) -> str:
        if key.startswith('blocks.'):
            key = key.removeprefix('blocks.')
            return (
                f'{i * 2 + 1}.'
                + self._convert_visual_encoder_layer_blocks(key)
            )
        if key.startswith('downsample.'):
            key = key.removeprefix('downsample.')
            return f'{i * 2 + 2}.{key}'
        raise ValueError(f"Unknown key: {key}")

    def _convert_visual_encoder_layers(self, key: str) -> str:
        index = key.index('.')
        prefix = int(key[:index])
        key = key[index + 1:]
        return 'features.' + self._convert_visual_encoder_layer(key, prefix)

    def _convert_visual_encoder(self, key: str) -> str:
        if key.startswith('patch_embed.'):
            key = key.removeprefix('patch_embed.')
            key = self._convert_visual_encoder_patch_embed(key)
        if key.startswith('layers.'):
            key = key.removeprefix('layers.')
            key = self._convert_visual_encoder_layers(key)
        return '_encoder.' + key

    def _convert(self, key: str) -> str | None:
        if key == 'logit_scale':
            return None
        if key.startswith('visual_encoder.'):
            key = key.removeprefix('visual_encoder.')
            return self._convert_visual_encoder(key)
        if key.startswith('image_proj.'):
            key = key.removeprefix('image_proj.')
            return f'_encoder.head.{key}'
        if key.startswith('wordvec_proj.'):
            key = key.removeprefix('wordvec_proj.')
            return f'_decoder._in_linear.{key}'
        if key.startswith('tagging_head.'):
            key = key.removeprefix('tagging_head.')
            return f'_decoder.{key}'
        if key.startswith('fc.'):
            key = key.removeprefix('fc.')
            return f'_decoder._out_linear.{key}'
        if key == 'label_embed':
            return '_category_embedding'
        return key


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
            names=list(Category.__dataclass_fields__.keys()),  # noqa: E501 pylint: disable=no-member
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


class Encoder(SwinTransformer):

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
            dim=1,
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

        embedding, *_ = super().forward(
            inputs_embeds=category_embedding,
            encoder_hidden_states=x,
        )

        logits = self._out_linear(embedding)
        logits = einops.rearrange(logits, '... 1 -> ...')  # TODO: specify ...
        return logits


class RAMplus(PretrainedMixin, nn.Module):
    STATE_DICT_CONVERTER = RAMplusStateDictConverterMixin

    NUM_DESCRIPTIONS = 51

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
                self.NUM_DESCRIPTIONS,
                hidden_channels,
            ),
        )
        self._decoder = Decoder(
            hidden_channels,
            decoder_depth,
            decoder_num_heads,
        )

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
