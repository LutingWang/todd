__all__ = [
    'Tokenizer',
]

import logging
import pathlib
from typing import Mapping
from typing_extensions import Self

import torch


class Tokenizer:
    TRANSLATION_TABLE = str.maketrans({
        ';': ',',
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
    })

    def __init__(self, text2token: Mapping[str, int]) -> None:
        self._text2token = dict(text2token)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Self:
        if isinstance(path, str):
            path = pathlib.Path(path)
        with path.open() as f:
            text2token = {line.strip(): i for i, line in enumerate(f)}
        return cls(text2token)

    def __len__(self) -> int:
        return len(self._text2token)

    def _encode(self, buffer: str) -> list[str]:
        if buffer == '':
            return []

        import jieba
        from pypinyin import Style, lazy_pinyin

        jieba.setLogLevel(logging.INFO)
        jieba.initialize()

        assert all(len(c.encode()) > 1 for c in buffer)
        segments = lazy_pinyin(
            jieba.cut(buffer),
            Style.TONE3,
            tone_sandhi=True,
        )

        for i in range(len(segments)):
            segments.insert(i * 2, ' ')
        segments.append(' ')

        return segments

    def _split(self, text: str) -> list[str]:
        segments: list[str] = []

        buffer = ''
        for c in text:
            if len(c.encode()) > 1:
                buffer += c
            else:
                segments.extend(self._encode(buffer))
                segments.append(c)
                buffer = ''
        segments.extend(self._encode(buffer))

        if segments[0] == ' ':
            segments.pop(0)
        if segments[-1] == ' ':
            segments.pop()
        assert len(segments) > 0

        return segments

    def __call__(self, text: str, duration: int) -> torch.Tensor:
        text = text.translate(self.TRANSLATION_TABLE)
        tokens = [
            self._text2token.get(segment, 0) for segment in self._split(text)
        ]
        assert len(tokens) < duration
        tokens += [-1] * duration
        return torch.tensor(tokens[:duration])
