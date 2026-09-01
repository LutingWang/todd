__all__ = [
    'Tokenizer',
]

import pathlib
from typing import Generator, Mapping
from typing_extensions import Self

import jieba
import torch
from pypinyin import Style, lazy_pinyin

from .constants import SEGMENT_PATTERN, TRANSLATION_TABLE


class Tokenizer:

    def __init__(self, character2token: Mapping[str, int]) -> None:
        self._character2token = dict(character2token)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Self:
        if isinstance(path, str):
            path = pathlib.Path(path)
        with path.open() as f:
            character2token = {line.strip(): i for i, line in enumerate(f)}
        return cls(character2token)

    def __len__(self) -> int:
        return len(self._character2token)

    def _parse(self, text: str) -> Generator[str, None, None]:
        segments = SEGMENT_PATTERN.findall(text)
        if segments[0].isascii():
            yield from segments.pop(0)
        for segment in segments:
            if segment.isascii():
                if len(segment) > 1:
                    yield ' '
                yield from segment
                continue
            words = jieba.cut(segment)
            pinyins = lazy_pinyin(words, Style.TONE3, tone_sandhi=True)
            assert len(segment) == len(pinyins)
            for character, pinyin in zip(segment, pinyins):
                if '\u3100' <= character <= '\u9fff':
                    yield ' '
                yield pinyin

    def __call__(self, text: str, duration: int) -> torch.Tensor:
        text = text.translate(TRANSLATION_TABLE)
        tokens = [
            self._character2token.get(character, 0)
            for character in self._parse(text)
        ]
        assert len(tokens) < duration
        tokens += [-1] * duration
        return torch.tensor(tokens[:duration])
