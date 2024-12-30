__all__ = [
    'TimeEmbedding',
]

import einops
import torch
from torch import nn

from todd.models.modules import PretrainedMixin, sinusoidal_position_embedding
from todd.utils import StateDictConverter


class TimeEmbeddingStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'time_mlp\.(.*)', r'_mlp.\1')


class TimeEmbedding(PretrainedMixin):
    STATE_DICT_CONVERTER = TimeEmbeddingStateDictConverter

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
