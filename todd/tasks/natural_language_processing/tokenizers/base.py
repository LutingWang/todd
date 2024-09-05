__all__ = [
    'BaseTokenizer',
]

from abc import abstractmethod
from typing import Iterable

import torch

from ..bpe import TokenSequence


class BaseTokenizer:
    SOS: str
    EOS: str

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

    def encodes(self, texts: Iterable[str], **kwargs) -> torch.Tensor:
        token_sequences = [
            self.encode(self.SOS + text + self.EOS, **kwargs) for text in texts
        ]
        tokens = torch.zeros(
            len(token_sequences),
            max(map(len, token_sequences)),
            dtype=torch.int,
        )
        for i, token_sequence in enumerate(token_sequences):
            tokens[i, :len(token_sequence)] = torch.tensor(token_sequence)
        return tokens
