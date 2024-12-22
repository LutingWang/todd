import argparse
import math
import pathlib
import re
from collections import UserDict
from typing import Any, Generator
from typing_extensions import Self

import einops
import torch
from pydub import silence
from transformers import AutomaticSpeechRecognitionPipeline, pipeline

import todd
from todd.configs import PyConfig
from todd.models.modules import F5_TTS, Vocos
from todd.models.modules.f5_tts import Tokenizer
from todd.patches.pydub import AudioSegment
from todd.utils import get_audio, init_seed

PUNCTUATION = '!,.:;?！，。：；？'
VOICE_PATTERN = re.compile(r'^\[(\w+)\]\s*')
PUNCTUATION_PATTERN = re.compile(rf'(?<=[{PUNCTUATION}])\s*')


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

        audio = einops.rearrange(audio, '1 t -> t')
        result = p(audio.numpy())
        return result['text']


class Voice:
    MAX_DURATION = 15

    def __init__(
        self,
        name: str,
        audio: torch.Tensor,
        sample_rate: int,
        transcription: str,
    ) -> None:
        self._name = name
        self._audio = audio.mean(0)
        self._sample_rate = sample_rate
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
    def load(
        cls,
        name: str,
        *,
        audio_file: Any,
        min_rms: float,
        sample_rate: int,
        transcription: str | None = None,
    ) -> Self:
        if isinstance(audio_file, str) and audio_file.startswith('http'):
            todd.logger.info("Downloading voice %s from %s", name, audio_file)
            audio_segment = AudioSegment.from_tensor(*get_audio(audio_file))
        else:
            todd.logger.info("Loading voice %s from %s", name, audio_file)
            audio_segment = AudioSegment.from_file(audio_file)

        audio_segment = sum(
            silence.split_on_silence(audio_segment, 1000, -50, 1000, 10),
            AudioSegment.empty(),
        )

        if audio_segment.duration_seconds > cls.MAX_DURATION:
            audio_segment = sum(
                silence.split_on_silence(audio_segment, 100, -40, 1000, 10),
                AudioSegment.empty(),
            )

        start = silence.detect_leading_silence(audio_segment, -42)
        end = silence.detect_leading_silence(audio_segment.reverse(), -42)
        assert start + end < len(audio_segment)
        audio_segment = audio_segment[start:-end]

        assert audio_segment.duration_seconds <= cls.MAX_DURATION
        audio_segment += AudioSegment.silent(50)

        audio_segment = audio_segment.clamp(min_rms)
        audio_segment = audio_segment.set_frame_rate(sample_rate)
        audio = audio_segment.to_tensor()

        if transcription is None:
            transcription = Whisper.transcript(audio)
            todd.logger.info("Transcription\n%s", transcription)

        transcription = transcription.strip()
        if transcription[-1] not in PUNCTUATION:
            transcription += (
                '. ' if len(transcription[-1].encode()) == 1 else '。'
            )

        return cls(name, audio, sample_rate, transcription)


class Voices(UserDict[str, Voice]):
    MAX_DURATION = 25

    @classmethod
    def load(cls, voices_file: pathlib.Path, **kwargs) -> Self:
        return cls({
            k: Voice.load(k, **v, **kwargs)
            for k, v in PyConfig.load(voices_file).items()
        })

    def _parse_voice(self, line: str) -> tuple[Voice, str]:
        match = VOICE_PATTERN.match(line)
        if match is None:
            voice = 'default'
        else:
            voice = match[1]
            line = line.removeprefix(match[0])
        return self[voice], line

    def _parse_chunks(
        self,
        line: str,
        max_bytes: float,
    ) -> Generator[str, None, None]:
        chunks = ''
        for chunk in PUNCTUATION_PATTERN.split(line):
            assert len(chunk.encode()) < max_bytes
            if len(chunks.encode()) + len(chunk.encode()) >= max_bytes:
                # chunks is not empty
                yield chunks
                chunks = ''
            if len(chunks) > 0 and len(chunks[-1].encode()) == 1:
                chunks += ' '
            chunks += chunk
        yield chunks

    def parse(
        self,
        lines_file: pathlib.Path,
    ) -> Generator[tuple[Voice, str], None, None]:
        with lines_file.open() as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                voice, line = self._parse_voice(line)
                max_bytes = (self.MAX_DURATION - voice.duration) * voice.bps
                for chunk in self._parse_chunks(line, max_bytes):
                    yield voice, chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('voices', type=pathlib.Path)
    parser.add_argument('lines', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--min-rms', type=float, default=0.1)
    parser.add_argument('--sample-rate', type=int, default=24_000)
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--remove-silence', action='store_true', default=False)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main() -> None:
    args = parse_args()

    init_seed(args.seed)

    voices = Voices.load(
        args.voices,
        min_rms=args.min_rms,
        sample_rate=args.sample_rate,
    )

    tokenizer = Tokenizer.load('pretrained/f5_tts/F5TTS_Base.txt')

    vocos = Vocos(sample_rate=args.sample_rate)
    f5_tts = F5_TTS(
        mel_channels=vocos.mel_channels,
        text_num_embeddings=len(tokenizer),
    )

    vocos.load_pretrained('pretrained/vocos/vocos-mel-24khz.pth')
    f5_tts.load_pretrained('pretrained/f5_tts/F5TTS_Base.pth')

    vocos.eval()
    f5_tts = f5_tts.eval()

    if todd.Store.cuda:  # pylint: disable=using-constant-test
        vocos = vocos.cuda()
        f5_tts = f5_tts.cuda()

    full_audio_segment = AudioSegment.empty()
    running_voice = None
    for voice, line in voices.parse(args.lines):
        todd.logger.info("Voice %s\n%s", voice._name, line)

        audio = voice.audio
        if todd.Store.cuda:  # pylint: disable=using-constant-test
            audio = audio.cuda()

        mel_spectrogram = vocos.encode(audio)
        _, t = mel_spectrogram.shape

        text = voice.transcription + line
        assert len(text) <= t

        ratio = len(line.encode()) / len(voice.transcription.encode())
        ratio = 1 + ratio / args.speed
        duration = math.ceil(t * ratio)

        tokens = tokenizer(text, duration)
        if todd.Store.cuda:  # pylint: disable=using-constant-test
            tokens = tokens.cuda()

        mel_spectrogram = f5_tts(mel_spectrogram[None], tokens[None], duration)
        mel_spectrogram = mel_spectrogram[..., t:]

        audio = vocos.decode(mel_spectrogram)
        audio_segment = AudioSegment.from_tensor(audio, voice.sample_rate)

        todd.logger.info("Duration %.3fs\n", audio_segment.duration_seconds)

        if voice == running_voice:
            full_audio_segment = full_audio_segment.append(audio_segment)
        else:
            full_audio_segment += audio_segment
        running_voice = voice

    if args.remove_silence:
        full_audio_segment = sum(
            silence.split_on_silence(full_audio_segment, 1000, -50, 500, 10),
            AudioSegment.empty(),
        )

    full_audio_segment = full_audio_segment.clamp(args.min_rms)

    output: pathlib.Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    full_audio_segment.export(output, 'wav')


if __name__ == '__main__':
    main()

# python docs/source/pretrained/f5_tts.py \
#     docs/source/pretrained/f5_tts/voices.py \
#     docs/source/pretrained/f5_tts/lines.txt \
#     f5_tts.wav \
#     --speed 1.2
#     --remove-silence \
