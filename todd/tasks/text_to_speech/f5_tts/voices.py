__all__ = [
    'Voice',
]

from todd.patches.pydub import AudioSegment

from ..voices import Voice as BaseVoice
from .constants import EN_PUNCTUATIONS, ZH_PUNCTUATIONS
from .utils import remove_long_silence, remove_short_silence, strip_silence


class Voice(BaseVoice):
    MAX_DURATION = 15

    def __init__(
        self,
        name: str,
        audio_segment: AudioSegment,
        *args,
        min_rms: float,
        **kwargs,
    ) -> None:
        audio_segment = remove_long_silence(audio_segment)
        if audio_segment.duration_seconds > self.MAX_DURATION:
            audio_segment = remove_short_silence(audio_segment)
        audio_segment = strip_silence(audio_segment)
        assert audio_segment.duration_seconds <= self.MAX_DURATION
        audio_segment += AudioSegment.silent(50)

        audio_segment = audio_segment.clamp(min_rms)

        super().__init__(name, audio_segment, *args, **kwargs)

        self._transcription = self._normalize_transcription(
            self._transcription,
        )

    def _normalize_transcription(self, transcription: str) -> str:
        transcription = transcription.strip()

        if transcription[-1].isascii():
            if transcription[-1] not in EN_PUNCTUATIONS:
                transcription += '.'
            transcription += ' '
            return transcription

        if transcription[-1] not in ZH_PUNCTUATIONS:
            transcription += '。'
        return transcription
