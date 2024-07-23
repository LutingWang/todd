__all__ = [
    'BPE',
    'Trainer',
]

from collections import Counter
from typing import Iterable

Token = int
TokenPair = tuple[Token, Token]
TokenSequence = list[Token]


def merge(
    token_sequence: TokenSequence,
    token_pair: TokenPair,
    token: Token,
) -> tuple[TokenSequence, set[int]]:
    token_pair_ = list(token_pair)

    new_token_sequence: TokenSequence = []
    indices: set[int] = set()

    i = 0
    while i < len(token_sequence):
        if token_sequence[i:i + 2] == token_pair_:
            indices.add(len(new_token_sequence))
            new_token_sequence.append(token)
            i += 2
        else:
            new_token_sequence.append(token_sequence[i])
            i += 1

    return new_token_sequence, indices


class BPE:

    def __init__(
        self,
        codebook_size: int,
        token_pairs: list[TokenPair],
    ) -> None:
        self._codebook_size = codebook_size
        self._token_pairs = token_pairs
        self._token_mappings = ([[i] for i in range(codebook_size)]
                                + [None] * len(token_pairs))

    def encode(self, token_sequence: TokenSequence) -> TokenSequence:
        while len(token_sequence) >= 2:
            token_pairs = set(zip(token_sequence, token_sequence[1:]))
            i = next(
                (
                    i for i, token_pair in enumerate(self._token_pairs)
                    if token_pair in token_pairs
                ),
                None,
            )
            if i is None:
                break
            token_sequence, _ = merge(
                token_sequence,
                self._token_pairs[i],
                self._codebook_size + i,
            )
        return token_sequence

    def _decode(self, token: Token) -> TokenSequence:
        token_sequence = self._token_mappings[token]
        if token_sequence is None:
            token_pair = self._token_pairs[token - self._codebook_size]
            token_sequence = (
                self._decode(token_pair[0]) + self._decode(token_pair[1])
            )
            self._token_mappings[token] = token_sequence
        return token_sequence

    def decode(self, token_sequence: TokenSequence) -> TokenSequence:
        return sum(map(self._decode, token_sequence), [])


class Trainer:

    def __init__(
        self,
        token_sequences: Iterable[TokenSequence],
        codebook_size: int,
        max_size: int,
    ) -> None:
        token_sequences = list(token_sequences)
        self._token_sequences = token_sequences

        self._codebook_size = codebook_size
        self._max_size = max_size

        counter: Counter[TokenPair] = Counter()
        for token_sequence in token_sequences:
            token_pairs = zip(token_sequence, token_sequence[1:])
            counter.update(token_pairs)
        self._counter = counter

    def merge(self, token_pair: TokenPair, token: Token) -> None:
        previous_tokens: Counter[Token] = Counter()
        next_tokens: Counter[Token] = Counter()

        for i, token_sequence in enumerate(self._token_sequences):
            token_sequence, indices = merge(
                token_sequence,
                token_pair,
                token,
            )
            self._token_sequences[i] = token_sequence
            for j in indices - {0}:
                previous_tokens[token_sequence[j - 1]] += 1
            for j in indices - {len(token_sequence) - 1}:
                next_tokens[token_sequence[j + 1]] += 1

        for previous_token, n in previous_tokens.items():
            self._counter[previous_token, token] += n
            self._counter[previous_token, token_pair[0]] -= n

        for next_token, n in next_tokens.items():
            self._counter[token, next_token] += n
            self._counter[token_pair[1], next_token] -= n

        self._counter.pop(token_pair)

    def train(self) -> BPE:
        token_pairs: list[TokenPair] = []
        for token in range(self._codebook_size, self._max_size):
            most_common = self._counter.most_common(1)
            if not most_common:
                break
            (token_pair, n), = most_common
            if n <= 1:
                break
            token_pairs.append(token_pair)
            self.merge(token_pair, token)
        return BPE(self._codebook_size, token_pairs)
