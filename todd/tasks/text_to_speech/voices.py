__all__ = [
    'Voice',
]

import pathlib
from typing import Any
from typing_extensions import Self

import einops
import torch
import torchaudio.functional as F
from transformers import AutomaticSpeechRecognitionPipeline, pipeline

import todd
from todd.configs import PyConfig
from todd.patches.pydub import AudioSegment
from todd.utils import get_audio

from .utils import normalize_text


class Whisper:
    _pipeline: AutomaticSpeechRecognitionPipeline | None = None

    @classmethod
    def transcript(cls, audio: torch.Tensor) -> str:
        p = cls._pipeline
        if p is None:
            p = pipeline(
                'automatic-speech-recognition',
                'pretrained/whisper/whisper-large-v3-turbo',
                torch_dtype='auto',
                device_map='auto',
            )
            cls._pipeline = p

        result = p(audio.numpy())
        return result['text']


class Voice:

    def __init__(
        self,
        name: str,
        audio_segment: AudioSegment,
        sample_rate: int,
        transcription: str | None = None,
    ) -> None:
        self._name = name

        audio = audio_segment.to_tensor()
        audio = F.resample(audio, audio_segment.frame_rate, sample_rate)
        audio = einops.reduce(audio, 'c t -> t', 'mean')
        self._audio = audio

        self._sample_rate = sample_rate

        if transcription is None:
            transcription = Whisper.transcript(audio)
            todd.logger.info("Transcription\n%s", transcription)
        transcription = normalize_text(transcription)
        self._transcription = transcription

    @property
    def audio(self) -> torch.Tensor:
        return self._audio

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def transcription(self) -> str:
        return self._transcription

    @property
    def num_samples(self) -> int:
        num_samples, = self._audio.shape
        return num_samples

    @property
    def duration(self) -> float:
        return self.num_samples / self._sample_rate

    @property
    def bps(self) -> float:
        return len(self._transcription.encode()) / self.duration

    @classmethod
    def load(cls, name: str, audio_file: Any, **kwargs) -> Self:
        if isinstance(audio_file, str) and audio_file.startswith('http'):
            todd.logger.info("Downloading voice %s from %s", name, audio_file)
            audio_segment = AudioSegment.from_tensor(*get_audio(audio_file))
        else:
            todd.logger.info("Loading voice %s from %s", name, audio_file)
            audio_segment = AudioSegment.from_file(audio_file)
        return cls(name, audio_segment, **kwargs)

    @classmethod
    def from_config(
        cls,
        config_file: pathlib.Path,
        **kwargs,
    ) -> dict[str, Self]:
        config = PyConfig.load(config_file)
        return {k: cls.load(k, **v, **kwargs) for k, v in config.items()}
