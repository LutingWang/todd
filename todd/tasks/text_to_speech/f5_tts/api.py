__all__ = [
    'F5_TTS',
]

import logging
import math
import pathlib
from typing import Generator

import jieba

from todd.loggers import logger
from todd.models.modules import Vocos
from todd.patches.py_ import remove_prefix
from todd.patches.pydub import AudioSegment
from todd.utils import Store

from .lines import Lines
from .modules import Model
from .tokenizer import Tokenizer
from .utils import remove_long_silence
from .voices import Voice

jieba.setLogLevel(logging.INFO)
jieba.initialize()


class F5_TTS:  # noqa: N801 pylint: disable=invalid-name

    def __init__(
        self,
        voices: pathlib.Path,
        min_rms: float = 0.1,
        sample_rate: int = 24_000,
    ) -> None:
        self._voices = Voice.from_config(
            voices,
            min_rms=min_rms,
            sample_rate=sample_rate,
        )
        self._min_rms = min_rms
        self._sample_rate = sample_rate

        tokenizer = Tokenizer.load('pretrained/f5_tts/F5TTS_Base.txt')
        self._tokenizer = tokenizer

        vocos = Vocos(sample_rate=sample_rate)
        self._vocos = vocos

        model = Model(
            mel_channels=vocos.mel_channels,
            text_num_embeddings=len(tokenizer),
        )
        self._model = model

        vocos.load_pretrained('pretrained/vocos/vocos-mel-24khz.pth')
        model.load_pretrained('pretrained/f5_tts/F5TTS_Base.pth')

        vocos.eval()
        model = model.eval()

        if Store.cuda:  # pylint: disable=using-constant-test
            vocos = vocos.cuda()
            model = model.cuda()

    def read(
        self,
        voice: Voice,
        line: str,
        speed: float = 1.0,
    ) -> AudioSegment:
        audio = voice.audio
        if Store.cuda:  # pylint: disable=using-constant-test
            audio = audio.cuda()

        mel_spectrogram = self._vocos.encode(audio)
        _, t = mel_spectrogram.shape

        text = voice.transcription + line
        assert len(text) <= t

        ratio = len(line.encode()) / len(voice.transcription.encode())
        ratio = 1 + ratio / speed
        duration = math.ceil(t * ratio)

        tokens = self._tokenizer(text, duration)
        if Store.cuda:  # pylint: disable=using-constant-test
            tokens = tokens.cuda()

        mel_spectrogram = self._model(
            mel_spectrogram[None],
            tokens[None],
            duration,
        )
        mel_spectrogram = mel_spectrogram[..., t:]

        audio = self._vocos.decode(mel_spectrogram)
        return AudioSegment.from_tensor(audio, voice.sample_rate)

    def read_file(
        self,
        lines_file: pathlib.Path,
        *args,
        **kwargs,
    ) -> Generator[tuple[Voice, AudioSegment], None, None]:
        lines = Lines(voices=self._voices).parse(lines_file)
        for voice, line in lines:
            logger.info("Voice %s\n%s", voice._name, line)
            audio_segment = self.read(voice, line, *args, **kwargs)
            logger.info("Duration %.3fs\n", audio_segment.duration_seconds)
            yield voice, audio_segment

    def run(
        self,
        *args,
        output_file: pathlib.Path,
        remove_silence: bool = False,
        **kwargs,
    ) -> None:
        stream = self.read_file(*args, **kwargs)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_format = remove_prefix(output_file.suffix, '.')

        full_audio_segment = AudioSegment.empty()
        running_voice = None
        for i, (voice, audio_segment) in enumerate(stream):
            audio_segment.export(
                output_file.with_stem(f'{output_file.stem}_{i}'),
                output_format,
            )

            if voice == running_voice:
                full_audio_segment = full_audio_segment.append(audio_segment)
            else:
                full_audio_segment += audio_segment
            running_voice = voice

        if remove_silence:
            full_audio_segment = remove_long_silence(full_audio_segment)

        full_audio_segment = full_audio_segment.clamp(self._min_rms)

        full_audio_segment.export(output_file, output_format)
