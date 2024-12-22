__all__ = [
    'F5_TTS',
]

import logging
import math
import pathlib
import re
from typing import Mapping, cast
from typing_extensions import Self

import einops
import torch
import torch.nn.functional as F
from torch import nn

from ...patches.torch import Sequential
from ...utils import StateDict, StateDictConverter
from ..norms import AdaptiveLayerNorm
from .position_embeddings import (
    rotary_position_embedding,
    sinusoidal_position_embedding,
)
from .pretrained import PretrainedMixin
from .transformer import mlp

MAX_DURATION = 4096


class Tokenizer:
    TRANSLATION_TABLE = str.maketrans({
        ';': ',',
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
    })

    def __init__(self, text2token: Mapping[str, int]) -> None:
        self._text2token = dict(text2token)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Self:
        if isinstance(path, str):
            path = pathlib.Path(path)
        with path.open() as f:
            text2token = {line.strip(): i for i, line in enumerate(f)}
        return cls(text2token)

    def __len__(self) -> int:
        return len(self._text2token)

    def _encode(self, buffer: str) -> list[str]:
        if buffer == '':
            return []

        import jieba
        from pypinyin import Style, lazy_pinyin

        jieba.setLogLevel(logging.INFO)
        jieba.initialize()

        assert all(len(c.encode()) > 1 for c in buffer)
        segments = lazy_pinyin(
            jieba.cut(buffer),
            Style.TONE3,
            tone_sandhi=True,
        )

        for i in range(len(segments)):
            segments.insert(i * 2, ' ')
        segments.append(' ')

        return segments

    def _split(self, text: str) -> list[str]:
        segments: list[str] = []

        buffer = ''
        for c in text:
            if len(c.encode()) > 1:
                buffer += c
            else:
                segments.extend(self._encode(buffer))
                segments.append(c)
                buffer = ''
        segments.extend(self._encode(buffer))

        if segments[0] == ' ':
            segments.pop(0)
        if segments[-1] == ' ':
            segments.pop()
        assert len(segments) > 0

        return segments

    def __call__(self, text: str, duration: int) -> torch.Tensor:
        text = text.translate(self.TRANSLATION_TABLE)
        tokens = [
            self._text2token.get(segment, 0) for segment in self._split(text)
        ]
        assert len(tokens) < duration
        tokens += [-1] * duration
        return torch.tensor(tokens[:duration])


class GRN(nn.Module):

    def __init__(self, *args, channels: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._weight = nn.Parameter(torch.empty(channels))
        self._bias = nn.Parameter(torch.empty(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_ = x.norm(dim=1, keepdim=True)
        norm = global_ / (global_.mean(-1, True) + 1e-6)
        return self._weight * (x * norm) + self._bias + x


class ConvNeXtV2Block(nn.Module):

    def __init__(self, *args, channels: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._in_conv = nn.Conv1d(
            channels,
            channels,
            7,
            padding=3,
            groups=channels,
        )
        self._norm = nn.LayerNorm(channels, eps=1e-6)
        self._proj1 = nn.Linear(channels, channels * 2)
        self._grn = GRN(channels=channels * 2)
        self._proj2 = nn.Linear(channels * 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = einops.rearrange(x, 'b t c -> b c t')
        x = self._in_conv(x)
        x = einops.rearrange(x, 'b c t -> b t c')
        x = self._norm(x)
        x = self._proj1(x)
        x = F.gelu(x)
        x = self._grn(x)
        x = self._proj2(x)
        return identity + x


class AudioEmbedding(nn.Module):

    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        duration: int,
    ) -> torch.Tensor:
        b, c, t = mel_spectrogram.shape
        embedding = mel_spectrogram.new_zeros(b, c, duration)
        embedding[..., :t] = mel_spectrogram.clamp_min(1e-5).log()
        embedding = einops.rearrange(embedding, 'b c t -> b t c')
        return embedding


class TextEmbedding(nn.Module):

    def __init__(
        self,
        *args,
        num_embeddings: int,
        embedding_dim: int = 512,
        depth: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._embedding = nn.Embedding(num_embeddings + 1, embedding_dim)

        position_embedding = sinusoidal_position_embedding(
            torch.arange(MAX_DURATION),
            embedding_dim + 2,
        )
        position_embedding = einops.rearrange(
            position_embedding,
            'l (d two) -> l two d',
            two=2,
        )
        position_embedding = position_embedding.flip(1)
        position_embedding = position_embedding[..., :-1]
        position_embedding = einops.rearrange(
            position_embedding,
            'l two d -> l (two d)',
            two=2,
        )
        self.position_embedding = position_embedding

        self._blocks = nn.Sequential(
            *[ConvNeXtV2Block(channels=embedding_dim) for _ in range(depth)],
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding.embedding_dim

    @property
    def position_embedding(self) -> torch.Tensor:
        return self.get_buffer('_position_embedding')

    @position_embedding.setter
    def position_embedding(self, value: torch.Tensor) -> None:
        self.register_buffer('_position_embedding', value, False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, t = tokens.shape
        position_embedding = self.position_embedding[:t]
        position_embedding = einops.rearrange(
            position_embedding,
            't c -> 1 t c',
        )

        embedding: torch.Tensor = self._embedding(tokens)
        embedding = embedding + position_embedding
        embedding = self._blocks(embedding)
        return embedding


class TimeEmbedding(nn.Module):

    def __init__(
        self,
        *args,
        hidden_dim: int = 256,
        embedding_dim: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._hidden_dim = hidden_dim
        self._embedding_dim = embedding_dim
        self._mlp = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, t: torch.Tensor, scale: int = 1000) -> torch.Tensor:
        t = t * scale
        embedding = sinusoidal_position_embedding(t, self._hidden_dim)
        embedding = einops.rearrange(
            embedding,
            't (d two) -> t (two d)',
            two=2,
        )
        embedding = self._mlp(embedding)
        return embedding


class Attention(nn.Module):

    def __init__(
        self,
        *args,
        channels: int,
        num_heads: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert channels % num_heads == 0
        self._head_channels = channels // num_heads

        self._in_proj = nn.Linear(channels, 3 * channels)
        self._out_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, _ = x.shape

        qkv: torch.Tensor = self._in_proj(x)
        qkv = einops.rearrange(
            qkv,
            'b t (three nh hc) -> three b nh t hc',
            three=3,
            hc=self._head_channels,
        )
        q, k, v = qkv.unbind()

        position_embedding = sinusoidal_position_embedding(
            torch.arange(t, device=x.device),
            self._head_channels + 2,
        )
        position_embedding = position_embedding[..., :-2]
        q[:, 0] = rotary_position_embedding(q[:, 0], position_embedding)
        k[:, 0] = rotary_position_embedding(k[:, 0], position_embedding)

        x = F.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, 'b nh t hc -> b t (nh hc)')

        x = self._out_proj(x)

        return x


class Block(nn.Module):

    def __init__(self, *args, channels: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gates = nn.Linear(channels, channels * 2)

        self._norm1 = AdaptiveLayerNorm(channels, 1e-6, condition_dim=channels)
        self._attention = Attention(channels=channels)

        self._norm2 = AdaptiveLayerNorm(channels, 1e-6, condition_dim=channels)
        self._mlp = mlp(channels, channels * 2, channels)
        cast(nn.GELU, self._mlp[1]).approximate = 'tanh'

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        gates: torch.Tensor = self._gates(condition)
        gate_attention, gate_mlp = gates.chunk(2, -1)

        norm = self._norm1(x, condition=condition)
        attention = self._attention(x=norm)
        x = x + gate_attention.unsqueeze(1) * attention

        norm = self._norm2(x, condition=condition)
        mlp_ = self._mlp(norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_

        return x


class DiT(nn.Module):

    def __init__(
        self,
        *args,
        in_channels: int,
        hidden_channels: int = 1024,
        out_channels: int,
        depth: int = 22,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        time_embedding = TimeEmbedding(embedding_dim=hidden_channels)
        self._time_embedding = time_embedding

        self._in_proj = nn.Linear(in_channels, hidden_channels)
        self._residual = nn.Sequential(
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                31,
                padding=15,
                groups=16,
            ),
            nn.Mish(),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                31,
                padding=15,
                groups=16,
            ),
            nn.Mish(),
        )

        blocks = [Block(channels=hidden_channels) for _ in range(depth)]
        self._blocks = Sequential(*blocks)

        self._out_norm = AdaptiveLayerNorm(
            hidden_channels,
            1e-6,
            condition_dim=hidden_channels,
        )
        self._out_proj = nn.Linear(hidden_channels, out_channels)

    @property
    def out_channels(self) -> int:
        return self._out_proj.out_features

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat((x, condition), -1)

        x = self._in_proj(x)

        identity = x
        x = einops.rearrange(x, 'b t c -> b c t')
        x = self._residual(x)
        x = einops.rearrange(x, 'b c t -> b t c')
        x = identity + x

        x = self._blocks(x, condition=time_embedding)

        x = self._out_norm(x, condition=time_embedding)
        x = self._out_proj(x)

        return x

    def sample(
        self,
        condition: torch.Tensor,
        steps: int = 32,
        cfg: float = 2.0,
    ) -> torch.Tensor:
        *shape, _ = condition.shape
        x = torch.randn(*shape, self.out_channels, device=condition.device)

        uncondition = torch.zeros_like(condition)

        t = torch.linspace(0, math.pi / 2, steps + 1, device=condition.device)
        t = 1 - t.cos()

        time_embedding = self._time_embedding(t)
        dt = t[1:] - t[:-1]

        for i in range(steps):
            time_embedding_ = time_embedding[[i]]
            conditional = self(x, condition, time_embedding_)
            unconditional = self(x, uncondition, time_embedding_)
            derivative = conditional + (conditional - unconditional) * cfg
            x = x + dt[i] * derivative

        return x


class F5_TTSStateDictConverter(  # noqa: N801 pylint: disable=invalid-name
    StateDictConverter,
):

    def load(self, *args, **kwargs) -> StateDict:
        checkpoint = super().load(*args, **kwargs)
        return cast(StateDict, checkpoint['ema_model_state_dict'])

    def _convert_transformer_text_embed_text_blocks_grn(self, key: str) -> str:
        if key == 'gamma':
            key = '_weight'
        elif key == 'beta':
            key = '_bias'
        else:
            raise ValueError(f"Unknown key: {key}")
        return f'_grn.{key}'

    def _convert_transformer_text_embed_text_blocks(self, key: str) -> str:
        index = key.index('.') + 1
        prefix = key[:index]
        key = key[index:]

        if key.startswith('dwconv.'):
            key = key.removeprefix('dwconv.')
            key = f'_in_conv.{key}'
        elif key.startswith('norm.'):
            key = f'_{key}'
        elif key.startswith('pwconv'):
            key = key.removeprefix('pwconv')
            key = f'_proj{key}'
        elif key.startswith('grn.'):
            key = key.removeprefix('grn.')
            key = self._convert_transformer_text_embed_text_blocks_grn(key)

        return f'_blocks.{prefix}{key}'

    def _convert_transformer_text_embed(self, key: str) -> str:
        if key.startswith('text_embed.'):
            key = key.removeprefix('text_embed.')
            return f'_embedding.{key}'
        if key.startswith('text_blocks.'):
            key = key.removeprefix('text_blocks.')
            return self._convert_transformer_text_embed_text_blocks(key)
        raise ValueError(f"Unknown key: {key}")

    def _convert_transformer_input_embed(self, key: str) -> str:
        if key.startswith('conv_pos_embed.'):
            key = key.removeprefix('conv_pos_embed.')
        if key.startswith('proj.'):
            key = key.removeprefix('proj.')
            key = f'_in_proj.{key}'
        if key.startswith('conv1d.'):
            key = key.removeprefix('conv1d.')
            key = f'_residual.{key}'
        return f'_dit.{key}'

    def _convert_transformer_blocks_attn(self, key: str) -> str:
        if key.startswith('to_out.0.'):
            key = key.removeprefix('to_out.0.')
            key = f'_out_proj.{key}'
        return f'_attention.{key}'

    def _convert_transformer_blocks_ff(self, key: str) -> str:
        assert key.startswith('ff.')
        key = key.removeprefix('ff.')
        if key.startswith('0.0.'):
            key = key.removeprefix('0.')
        return f'_mlp.{key}'

    def _convert_transformer_blocks(self, key: str) -> str:
        index = key.index('.') + 1
        prefix = key[:index]
        key = key[index:]

        if key.startswith('ff.'):
            key = key.removeprefix('ff.')
            key = self._convert_transformer_blocks_ff(key)
        elif key.startswith('attn_norm.'):
            key = key.removeprefix('attn_norm.')
            key = f'_norm1._{key}'
        elif key.startswith('attn.'):
            key = key.removeprefix('attn.')
            key = self._convert_transformer_blocks_attn(key)
        elif key.startswith('ff_norm.'):
            key = key.removeprefix('ff_norm.')
            key = f'_norm2._{key}'

        return f'_dit._blocks.{prefix}{key}'

    def _convert_transformer(self, key: str) -> str | None:
        if key.startswith('time_embed.time_mlp.'):
            key = key.removeprefix('time_embed.time_mlp.')
            return f'_dit._time_embedding._mlp.{key}'
        if key.startswith('text_embed.'):
            key = key.removeprefix('text_embed.')
            key = self._convert_transformer_text_embed(key)
            return f'_text_embedding.{key}'
        if key.startswith('input_embed.'):
            key = key.removeprefix('input_embed.')
            return self._convert_transformer_input_embed(key)
        if key.startswith('transformer_blocks.'):
            key = key.removeprefix('transformer_blocks.')
            return self._convert_transformer_blocks(key)
        if key.startswith('norm_out.'):
            key = key.removeprefix('norm_out.')
            return f'_dit._out_norm._{key}'
        if key.startswith('proj_out.'):
            key = key.removeprefix('proj_out.')
            return f'_dit._out_proj.{key}'
        if key.startswith('rotary_embed.'):
            return None
        return f'_dit.{key}'

    def _convert(self, key: str) -> str | None:
        if key in ['initted', 'step']:
            return None
        assert key.startswith('ema_model.')
        key = key.removeprefix('ema_model.')
        if key.startswith('mel_spec.'):
            return None
        if key.startswith('transformer.'):
            key = key.removeprefix('transformer.')
            return self._convert_transformer(key)
        raise ValueError(f"Unknown key: {key}")

    def _post_convert(self, state_dict: StateDict) -> StateDict:
        pattern = re.compile(
            r'^(_dit\._blocks\.\d+\.)_norm1\._linear\.(.*)$',
        )
        update: StateDict = dict()
        for k, v in state_dict.items():
            match = pattern.match(k)
            if match is None:
                continue
            bias1, weight1, gate1, bias2, weight2, gate2 = v.chunk(6)
            state_dict[k] = torch.cat((weight1, bias1))
            update[f'{match[1]}_norm2._linear.{match[2]}'] = torch.cat(
                (weight2, bias2),
            )
            update[f'{match[1]}_gates.{match[2]}'] = torch.cat((gate1, gate2))
        state_dict.update(update)

        for i in range(22):
            prefix = f'_dit._blocks.{i}._attention.'

            q_weight = state_dict.pop(prefix + 'to_q.weight')
            k_weight = state_dict.pop(prefix + 'to_k.weight')
            v_weight = state_dict.pop(prefix + 'to_v.weight')
            weight = torch.cat((q_weight, k_weight, v_weight))
            state_dict[prefix + '_in_proj.weight'] = weight

            q_bias = state_dict.pop(prefix + 'to_q.bias')
            k_bias = state_dict.pop(prefix + 'to_k.bias')
            v_bias = state_dict.pop(prefix + 'to_v.bias')
            bias = torch.cat((q_bias, k_bias, v_bias))
            state_dict[prefix + '_in_proj.bias'] = bias

        for i in range(4):
            prefix = f'_text_embedding._blocks.{i}._grn.'
            weight_ = prefix + '_weight'
            state_dict[weight_] = state_dict[weight_].flatten()
            bias_ = prefix + '_bias'
            state_dict[bias_] = state_dict[bias_].flatten()

        return super()._post_convert(state_dict)


class F5_TTS(PretrainedMixin):  # noqa: N801 pylint: disable=invalid-name
    STATE_DICT_CONVERTER = F5_TTSStateDictConverter

    def __init__(
        self,
        *args,
        mel_channels: int,
        text_num_embeddings: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        audio_embedding = AudioEmbedding()
        self._audio_embedding = audio_embedding

        text_embedding = TextEmbedding(num_embeddings=text_num_embeddings)
        self._text_embedding = text_embedding

        self._dit = DiT(
            in_channels=mel_channels * 2 + text_embedding.embedding_dim,
            out_channels=mel_channels,
        )

    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        tokens: torch.Tensor,
        duration: int,
    ) -> torch.Tensor:
        assert duration <= MAX_DURATION

        audio_embedding = self._audio_embedding(mel_spectrogram, duration)
        text_embedding = self._text_embedding(tokens + 1)
        condition = torch.cat((audio_embedding, text_embedding), -1)

        y = self._dit.sample(condition)

        return einops.rearrange(y, 'b t c -> b c t')
