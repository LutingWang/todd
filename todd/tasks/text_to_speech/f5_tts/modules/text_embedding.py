__all__ = [
    'TextEmbedding',
]

import einops
import torch
import torch.nn.functional as F
from torch import nn

from todd.models.modules import PretrainedMixin, sinusoidal_position_embedding
from todd.models.modules.vocos import BaseConvNeXt
from todd.utils import StateDict, StateDictConverter
from todd.utils.state_dicts import SequentialStateDictConverterMixin

from ..constants import MAX_DURATION


class GRN(nn.Module):

    def __init__(self, *args, channels: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._weight = nn.Parameter(torch.empty(channels))
        self._bias = nn.Parameter(torch.empty(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_ = x.norm(dim=1, keepdim=True)
        norm = global_ / (global_.mean(-1, True) + 1e-6)
        return self._weight * (x * norm) + self._bias + x


class ConvNeXtV2StateDictConverter(
    SequentialStateDictConverterMixin,
    StateDictConverter,
):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'dwconv\.(.*)', r'_in_conv.\1')
        self._register_regex_converter(r'norm\..*', r'_\g<0>')
        self._register_regex_converter(r'pwconv(1|2)\.(.*)', r'_proj\1.\2')
        self._register_key_mapping('grn.gamma', '_grn._weight')
        self._register_key_mapping('grn.beta', '_grn._bias')

    def _post_convert(self, state_dict: StateDict) -> StateDict:
        state_dict['_grn._weight'] = state_dict['_grn._weight'].flatten()
        state_dict['_grn._bias'] = state_dict['_grn._bias'].flatten()
        return super()._post_convert(state_dict)


class ConvNeXtV2(BaseConvNeXt):
    STATE_DICT_CONVERTER = ConvNeXtV2StateDictConverter

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._proj1 = nn.Linear(self._channels, self._channels * 2)
        self._grn = GRN(channels=self._channels * 2)
        self._proj2 = nn.Linear(self._channels * 2, self._channels)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._proj1(x)
        x = F.gelu(x)
        x = self._grn(x)
        x = self._proj2(x)
        return x


class TextEmbeddingStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'text_embed\.(.*)', r'_embedding.\1')
        self._register_child_converter(
            'text_blocks',
            '_blocks',
            ConvNeXtV2StateDictConverter,
        )


class TextEmbedding(PretrainedMixin):
    STATE_DICT_CONVERTER = TextEmbeddingStateDictConverter

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
            *[ConvNeXtV2(channels=embedding_dim) for _ in range(depth)],
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
