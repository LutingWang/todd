__all__ = [
    'DiT',
]

import math
from typing import cast

import einops
import torch
import torch.nn.functional as F
from torch import nn

from todd.models import AdaptiveLayerNorm
from todd.models.modules import (
    PretrainedMixin,
    rotary_position_embedding,
    sinusoidal_position_embedding,
)
from todd.models.modules.transformer import mlp
from todd.patches.torch import Sequential
from todd.utils import StateDict, StateDictConverter, set_temp
from todd.utils.state_dicts import parallel_conversion

from .time_embedding import TimeEmbedding


class AttentionStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'to_out\.0\.(.*)', r'_out_proj.\1')

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        state_dict = super()._pre_convert(state_dict)
        for key in ('weight', 'bias'):
            state_dict[f'_in_proj.{key}'] = torch.cat((
                state_dict.pop(f'to_q.{key}'),
                state_dict.pop(f'to_k.{key}'),
                state_dict.pop(f'to_v.{key}'),
            ))
        return state_dict

    def _convert(self, key: str) -> str | None:
        if key.startswith('_in_proj.'):
            return key
        return super()._convert(key)


class Attention(PretrainedMixin):
    STATE_DICT_CONVERTER = AttentionStateDictConverter

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


class BlockStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_child_converter(
            'attn',
            '_attention',
            AttentionStateDictConverter,
        )
        self._register_regex_converter(r'ff\.ff\.0\.0\.(.*)', r'_mlp.0.\1')
        self._register_regex_converter(r'ff\.ff\.2\.(.*)', r'_mlp.2.\1')

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        state_dict = super()._pre_convert(state_dict)

        for key in ('weight', 'bias'):
            value = state_dict.pop(f'attn_norm.linear.{key}')
            bias1, weight1, gate_attention, bias2, weight2, gate_mlp = \
                value.chunk(6)
            state_dict[f'_gates.{key}'] = torch.cat((gate_attention, gate_mlp))
            state_dict[f'_norm1._linear.{key}'] = torch.cat((weight1, bias1))
            state_dict[f'_norm2._linear.{key}'] = torch.cat((weight2, bias2))

        return state_dict

    def _convert(self, key: str) -> str | None:
        if key.startswith(('_gates.', '_norm1.', '_norm2.')):
            return key
        return super()._convert(key)

    @parallel_conversion
    def convert(self, state_dict: StateDict, prefix: str) -> StateDict:  # noqa: E501 pylint: disable=arguments-differ
        module = cast(Sequential, self._module)
        with set_temp(self, '._module', module[int(prefix)]):
            return super().convert(state_dict)


class Block(PretrainedMixin):
    STATE_DICT_CONVERTER = BlockStateDictConverter

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


class DiTStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_child_converter(
            'time_embed',
            '_time_embedding',
            TimeEmbedding.STATE_DICT_CONVERTER,
        )
        self._register_regex_converter(
            r'input_embed\.proj\.(.*)',
            r'_in_proj.\1',
        )
        self._register_regex_converter(
            r'input_embed\.conv_pos_embed\.conv1d\.(.*)',
            r'_residual.\1',
        )
        self._register_child_converter(
            'transformer_blocks',
            '_blocks',
            BlockStateDictConverter,
        )
        self._register_regex_converter(
            r'norm_out\.(.*)',
            r'_out_norm._\1',
        )
        self._register_regex_converter(
            r'proj_out\.(.*)',
            r'_out_proj.\1',
        )

        self._register_regex_converter(r'rotary_embed\..*', None)


class DiT(PretrainedMixin):
    STATE_DICT_CONVERTER = DiTStateDictConverter

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
