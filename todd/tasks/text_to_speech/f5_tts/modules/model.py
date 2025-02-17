__all__ = [
    'Model',
]

from typing import cast

import einops
import torch

from todd.models.modules import PretrainedMixin
from todd.patches.py_ import remove_prefix
from todd.utils import StateDict, StateDictConverter

from ..constants import MAX_DURATION
from .audio_embedding import AudioEmbedding
from .dit import DiT
from .text_embedding import TextEmbedding


class ModelStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # to be called manually
        self._register_child_converter(
            'text_embed',
            '_text_embedding',
            TextEmbedding.STATE_DICT_CONVERTER,
        )

        self._register_child_converter(
            'transformer',
            '_dit',
            DiT.STATE_DICT_CONVERTER,
        )

        self._register_regex_converter(r'mel_spec\..*', None)

    def load(self, *args, **kwargs) -> StateDict:
        checkpoint = super().load(*args, **kwargs)
        return cast(StateDict, checkpoint['ema_model_state_dict'])

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        state_dict = super()._pre_convert(state_dict)
        state_dict.pop('initted')
        state_dict.pop('step')
        state_dict = {
            remove_prefix(k, 'ema_model.'): v
            for k, v in state_dict.items()
        }

        # convert text embedding state dict
        child_name = 'text_embed'
        child_prefix = f'transformer.{child_name}.'
        child_state_dict: StateDict = dict()
        new_state_dict: StateDict = dict()
        for key, value in state_dict.items():
            if key.startswith(child_prefix):
                child_state_dict[key.removeprefix(child_prefix)] = value
            else:
                new_state_dict[key] = value
        new_state_dict |= self._convert_child(child_name, child_state_dict)

        return new_state_dict

    def _convert(self, key: str) -> str | None:
        if key.startswith('_text_embedding.'):
            return key
        return super()._convert(key)


class Model(PretrainedMixin):
    STATE_DICT_CONVERTER = ModelStateDictConverter

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
