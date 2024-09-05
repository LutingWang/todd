__all__ = [
    'Transformer',
]

from typing import cast

import torch
from torch import nn

from ...bases.configs import Config
from ...patches.torch import Sequential
from ...registries import InitWeightsMixin


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
        self._norm1 = nn.LayerNorm(width, 1e-6)
        self._attention = nn.MultiheadAttention(
            width,
            num_heads,
            batch_first=True,
        )
        self._norm2 = nn.LayerNorm(width, 1e-6)
        self._mlp = mlp(width, int(width * mlp_ratio), width)

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
        x = x + attention
        norm = self._norm2(x)
        mlp_ = self._mlp(norm)
        x = x + mlp_
        return x


class Transformer(InitWeightsMixin, nn.Module):
    BLOCK_TYPE = Block

    def __init__(
        self,
        *args,
        max_length: int = 77,
        num_embeddings: int = 49408,
        width: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._token_embedding = nn.Embedding(num_embeddings, width)
        self._position_embedding = nn.Parameter(torch.empty(max_length, width))
        self._blocks = Sequential(
            *[
                self.BLOCK_TYPE(width=width, num_heads=num_heads)
                for _ in range(depth)
            ],
        )
        self._norm = nn.LayerNorm(width)

        attention_mask = torch.full((max_length, max_length), float('-inf'))
        attention_mask.triu_(1)
        self.attention_mask = attention_mask

    @property
    def width(self) -> int:
        return self._token_embedding.embedding_dim

    @property
    def depth(self) -> int:
        return len(self._blocks)

    @property
    def attention_mask(self) -> torch.Tensor:
        return self.get_buffer('_attention_mask')

    @attention_mask.setter
    def attention_mask(self, value: torch.Tensor) -> None:
        self.register_buffer('_attention_mask', value, False)

    def init_weights(self, config: Config) -> bool:
        super().init_weights(config)
        nn.init.normal_(self._token_embedding.weight, std=0.02)
        nn.init.normal_(self._position_embedding, std=0.01)

        std1 = self.width**-0.5
        std2 = (2 * self.width)**-0.5
        std3 = (2 * self.width * self.depth)**-0.5

        block: Block
        for block in self._blocks:
            nn.init.normal_(block._attention.in_proj_weight, std=std1)
            nn.init.normal_(block._attention.out_proj.weight, std=std3)
            nn.init.normal_(cast(nn.Linear, block._mlp[0]).weight, std=std2)
            nn.init.normal_(cast(nn.Linear, block._mlp[2]).weight, std=std3)

        return False

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        length = text.shape[1]

        x: torch.Tensor = self._token_embedding(text)

        position_embedding = self._position_embedding[:length]
        attention_mask = self.attention_mask[:length, :length]

        x = x + position_embedding
        x = self._blocks(x, attention_mask=attention_mask)
        x = self._norm(x)
        return x
