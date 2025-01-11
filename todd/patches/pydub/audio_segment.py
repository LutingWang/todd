__all__ = [
    'AudioSegment',
]

import io
from typing_extensions import Self

import torch
import torchaudio
from pydub import AudioSegment as BaseAudioSegment
from pydub.utils import ratio_to_db


class AudioSegment(BaseAudioSegment):

    def to_tensor(self) -> torch.Tensor:
        bytes_io = io.BytesIO()
        self.export(bytes_io, 'wav')
        audio, _ = torchaudio.load(bytes_io)
        return audio

    @classmethod
    def from_tensor(cls, audio: torch.Tensor, sample_rate: int) -> Self:
        bytes_io = io.BytesIO()
        torchaudio.save(bytes_io, audio.cpu(), sample_rate, format='wav')
        return cls.from_wav(bytes_io)

    def clamp(self, min_rms: float) -> Self:
        gain = min_rms * self.max_possible_amplitude / self.rms
        if gain > 1:
            self += ratio_to_db(gain)
        return self
