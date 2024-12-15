__all__ = [
    'Token',
    'TokenPair',
    'TokenSequence',
    'BaseTokenizer',
]

from abc import abstractmethod
from typing import Iterable, Mapping

import torch

from todd.patches.py_ import classproperty

Token = int
TokenPair = tuple[Token, Token]
TokenSequence = list[Token]


class BaseTokenizer:
    SOS: str
    EOS: str

    def __init__(
        self,
        *args,
        text2token: Mapping[str, Token],
        special_text2token: Mapping[str, Token],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._text2token = text2token
        self._token2text = {v: k for k, v in text2token.items()}
        self._special_text2token = special_text2token
        self._special_token2text = {
            v: k
            for k, v in special_text2token.items()
        }

    @classproperty
    def special_texts(self) -> list[str]:
        return [self.SOS, self.EOS]

    @abstractmethod
    def encode(
        self,
        text: str,
        *,
        max_length: int | None = None,
    ) -> TokenSequence:
        pass

    def encodes(
        self,
        texts: Iterable[str],
        *,
        max_length: int | None = None,
    ) -> torch.Tensor:
        sos_token = self._special_text2token[self.SOS]
        eos_token = self._special_text2token[self.EOS]
        token_sequences = [[
            sos_token,
            *self.encode(
                text,
                max_length=None if max_length is None else max_length - 2,
            ),
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

    def _token_to_text(self, token: Token) -> str:
        if token in self._special_token2text:
            return self._special_token2text[token]
        return self._token2text[token]

    def decode(self, token_sequence: TokenSequence) -> str:
        return ''.join(self._token_to_text(token) for token in token_sequence)
