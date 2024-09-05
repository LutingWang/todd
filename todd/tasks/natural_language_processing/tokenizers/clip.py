__all__ = [
    'CLIPTokenizer',
]

import gzip
import html
import itertools
import pathlib
from typing import Any

import regex as re

from ..bpe import BPE, TokenSequence
from .base import BaseTokenizer


class CLIPTokenizer(BaseTokenizer):
    SOS = '<|startoftext|>'
    EOS = '<|endoftext|>'

    EOW = '</w>'

    def __init__(self, *args, bpe_path: Any = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        byte2unicode = list(range(256))

        for i, byte in enumerate(
            itertools.chain(
                range(ord('!')),
                range(ord('~') + 1, ord('¡')),
                range(ord('¬') + 1, ord('®')),
            ),
            256,
        ):
            byte2unicode[byte] = i

        self._byte2unicode = list(map(chr, byte2unicode))
        self._unicode2byte = {
            unicode: byte
            for byte, unicode in enumerate(self._byte2unicode)
        }

        word2token = {
            unicode: byte
            for byte, unicode in enumerate(sorted(self._byte2unicode))
        }
        word2token.update({
            unicode + self.EOW: byte + len(word2token)
            for unicode, byte in word2token.items()
        })

        token_pairs = []
        if bpe_path is None:
            bpe_path = pathlib.Path(__file__).parent / 'clip_bpe.txt.gz'
        with gzip.open(bpe_path, 'rt', encoding='utf-8') as f:
            f.readline()  # skip first line
            for _ in range(49152 - 256 - 2):
                word1, word2 = f.readline().split()
                token_pairs.append((word2token[word1], word2token[word2]))
                word2token[word1 + word2] = len(word2token)

        self._word2token = word2token
        self._token2word = {v: k for k, v in word2token.items()}

        special_words = [self.SOS, self.EOS]
        special_word2token = {
            word: len(word2token) + i
            for i, word in enumerate(special_words)
        }
        self._special_word2token = special_word2token

        self._bpe = BPE(len(self._byte2unicode) * 2, token_pairs)

        self._word2token_sequence: dict[str, TokenSequence] = dict()  # cache
        self._word_pattern = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d"
            r"|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            re.IGNORECASE,
        )

    def _parse(self, text: str) -> list[str]:
        import ftfy
        text = ftfy.fix_text(text)
        text = html.unescape(text)
        text = html.unescape(text)  # do this a second time
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        return self._word_pattern.findall(text)

    def _encode(self, word: str) -> TokenSequence:
        if word in self._special_word2token:
            return [self._special_word2token[word]]

        word_sequence = [
            self._byte2unicode[byte] for byte in word.encode('utf-8')
        ]
        word_sequence[-1] = word_sequence[-1] + self.EOW
        token_sequence = [
            self._word2token[unicode] for unicode in word_sequence
        ]
        return self._bpe.encode(token_sequence)

    def encode(
        self,
        text: str,
        *,
        max_length: int | None = None,
    ) -> TokenSequence:
        token_sequence: TokenSequence = []
        for word in self._parse(text):
            word_token_sequence = self._word2token_sequence.get(word)
            if word_token_sequence is None:
                word_token_sequence = self._encode(word)
                self._word2token_sequence[word] = word_token_sequence
            token_sequence.extend(word_token_sequence)
        if max_length is not None:
            token_sequence = token_sequence[:max_length]
        return token_sequence

    def decode(self, token_sequence: TokenSequence) -> str:
        text = ''.join(self._token2word[token] for token in token_sequence)
        text = bytearray(self._unicode2byte[c] for c in text).decode('utf-8')
        text = text.replace(self.EOW, ' ')
        return text
