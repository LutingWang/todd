__all__ = [
    'Lines',
]

from typing import Generator, Iterable, cast

import jieba

import todd.tasks.natural_language_processing as nlp

from ..lines import Lines as BaseLines
from .constants import PUNCTUATION_PATTERN, WHITESPACE_PATTERN
from .voices import Voice


class Segmentor(nlp.segmentors.RegexSegmentor):

    def __init__(self, *args, max_bytes: int | None = None, **kwargs) -> None:
        super().__init__(*args, regex=PUNCTUATION_PATTERN, **kwargs)
        self._max_bytes = max_bytes

    def _is_valid(self, text: str) -> bool:
        return super()._is_valid(text) and (
            self._max_bytes is None or len(text.encode()) <= self._max_bytes
        )

    def _segment(self, text: str) -> Iterable[str]:
        for segment in super()._segment(text):
            if self._is_valid(segment):
                yield segment
            elif segment.isascii():
                yield from WHITESPACE_PATTERN.split(segment)
            else:
                yield from jieba.cut(segment)


class Lines(BaseLines):
    MAX_DURATION = 25

    def parse(
        self,
        *args,
        **kwargs,
    ) -> Generator[tuple[Voice, str], None, None]:
        for voice, line in super().parse(*args, **kwargs):
            voice = cast(Voice, voice)
            max_bytes = int((self.MAX_DURATION - voice.duration) * voice.bps)
            segmentor = Segmentor(max_bytes=max_bytes)
            for segment in segmentor.segment(line):
                yield voice, segment
