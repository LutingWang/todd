__all__ = [
    'BaseTokenizer',
]

from abc import abstractmethod
from typing import Iterable, Mapping

import torch

from ..bpe import TokenSequence


class BaseTokenizer:
    SOS: str
    EOS: str

    def __init__(
        self,
        *args,
        word2token: Mapping[str, int],
        special_word2token: Mapping[str, int],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._word2token = word2token
        self._token2word = {v: k for k, v in word2token.items()}
        self._special_word2token = special_word2token
        self._special_token2word = {
            v: k
            for k, v in special_word2token.items()
        }

    @abstractmethod
    def encode(
        self,
        text: str,
        *,
        max_length: int | None = None,
    ) -> TokenSequence:
        pass

    @abstractmethod
    def decode(self, token_sequence: TokenSequence) -> str:
        pass

    def word_to_token(self, word: str) -> int:
        if word in self._special_word2token:
            return self._special_word2token[word]
        return self._word2token[word]

    def token_to_word(self, token: int) -> str:
        if token in self._special_token2word:
            return self._special_token2word[token]
        return self._token2word[token]

    def encodes(
        self,
        texts: Iterable[str],
        *,
        max_length: int | None = None,
    ) -> torch.Tensor:
        if max_length is not None:
            max_length -= 2
        sos_token = self._special_word2token[self.SOS]
        eos_token = self._special_word2token[self.EOS]
        token_sequences = [[
            sos_token,
            *self.encode(text, max_length=max_length),
            eos_token,
        ] for text in texts]
        tokens = torch.zeros(
            len(token_sequences),
            max(map(len, token_sequences)),
            dtype=torch.int,
        )
        for i, token_sequence in enumerate(token_sequences):
            tokens[i, :len(token_sequence)] = torch.tensor(token_sequence)
        return tokens
