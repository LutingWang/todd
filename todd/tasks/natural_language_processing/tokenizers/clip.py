__all__ = [
    'CLIPTokenizer',
]

import gzip
import html
import itertools
import pathlib
from typing import Any

import ftfy
import regex as re

from ..bpe import BPE
from .base import BaseTokenizer, TokenSequence

UTF_8 = 'utf-8'


class Codec:
    NUM_CODES = 256

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        byte2ord = list(range(self.NUM_CODES))

        for i, byte in enumerate(
            itertools.chain(
                range(ord('!')),
                range(ord('~') + 1, ord('¡')),
                range(ord('¬') + 1, ord('®')),
            ),
            self.NUM_CODES,
        ):
            byte2ord[byte] = i

        byte2unicode = list(map(chr, byte2ord))
        unicode2byte = {
            unicode: byte
            for byte, unicode in enumerate(byte2unicode)
        }

        self._byte2unicode = byte2unicode
        self._unicode2byte = unicode2byte

    @property
    def codes(self) -> list[str]:
        return sorted(self._byte2unicode)

    def encode(self, text: str) -> str:
        return ''.join(self._byte2unicode[byte] for byte in text.encode(UTF_8))

    def decode(self, encoded_text: str) -> str:
        bytes_ = bytes(self._unicode2byte[unicode] for unicode in encoded_text)
        return bytes_.decode(UTF_8)


def load_bpe(path: Any, size: int) -> list[tuple[str, str]]:
    if path is None:
        path = pathlib.Path(__file__).parent / 'clip_bpe.txt.gz'
    bpe: list[tuple[str, str]] = []
    with gzip.open(path, 'rt', encoding=UTF_8) as f:
        f.readline()  # skip first line
        for _ in range(size):
            text1, text2 = f.readline().split()  # ensure length is 2
            bpe.append((text1, text2))
    return bpe


class Parser:
    PATTERN = '|'.join([
        r'<\|startoftext\|>',
        r'<\|endoftext\|>',
        "'s",
        "'t",
        "'re",
        "'ve",
        "'m"
        "'ll"
        "'d",
        r'[\p{L}]+',
        r'[\p{N}]',
        r'[^\s\p{L}\p{N}]+',
    ])

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pattern = re.compile(self.PATTERN, re.IGNORECASE)

    def __call__(self, text: str) -> list[str]:
        text = ftfy.fix_text(text)
        text = html.unescape(text)
        text = html.unescape(text)  # do this a second time
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        return self._pattern.findall(text)


class Cache:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._text2token_sequence: dict[str, TokenSequence] = dict()

    def get(self, text: str) -> TokenSequence | None:
        return self._text2token_sequence.get(text)

    def set_(self, text: str, token_sequence: TokenSequence) -> None:
        self._text2token_sequence[text] = token_sequence


class CLIPTokenizer(BaseTokenizer):
    SOS = '<|startoftext|>'
    EOS = '<|endoftext|>'

    EOW = '</w>'  # not a special token

    def __init__(self, *args, bpe_path: Any = None, **kwargs) -> None:
        codec = Codec()
        bpe = load_bpe(
            bpe_path,
            3 * 2**14 - codec.NUM_CODES - len(self.special_texts),
        )

        text2token = {
            unicode: byte
            for byte, unicode in enumerate(codec.codes)
        }
        text2token.update({
            unicode + self.EOW: byte + codec.NUM_CODES
            for unicode, byte in text2token.items()
        })
        text2token.update({
            text1 + text2: i
            for i, (text1, text2) in enumerate(bpe, len(text2token))
        })

        special_text2token = {
            text: len(text2token) + i
            for i, text in enumerate(self.special_texts)
        }

        super().__init__(
            *args,
            text2token=text2token,
            special_text2token=special_text2token,
            **kwargs,
        )

        self._codec = codec
        self._parser = Parser()
        self._cache = Cache()
        self._bpe = BPE(
            codec.NUM_CODES * 2,
            [(text2token[text1], text2token[text2]) for text1, text2 in bpe],
        )

    def _encode(self, text: str) -> TokenSequence:
        token_sequence = self._cache.get(text)
        if token_sequence is not None:
            return token_sequence

        if text in self._special_text2token:
            token_sequence = [self._special_text2token[text]]
        else:
            code_sequence = list(self._codec.encode(text))
            code_sequence[-1] = code_sequence[-1] + self.EOW
            token_sequence = [
                self._text2token[unicode] for unicode in code_sequence
            ]
            token_sequence = self._bpe.encode(token_sequence)

        self._cache.set_(text, token_sequence)
        return token_sequence

    def encode(
        self,
        text: str,
        *,
        max_length: int | None = None,
    ) -> TokenSequence:
        token_sequence = sum(map(self._encode, self._parser(text)), [])
        if max_length is not None:
            token_sequence = token_sequence[:max_length]
        return token_sequence

    def decode(self, token_sequence: TokenSequence) -> str:
        encoded_text = super().decode(token_sequence)
        return self._codec.decode(encoded_text).replace(self.EOW, ' ')
