__all__ = [
    'BPE',
    'BPETrainer',
]

import os
from collections import Counter
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Iterable, TypeVar

import tqdm
from torch import nn

from todd.loggers import logger
from todd.utils import ArgsKwargs, SerializeMixin

from .tokenizers import Token, TokenPair, TokenSequence

T = TypeVar('T', bound=nn.Module)


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
        try:
            j = token_sequence.index(token_pair_[0], i)
        except ValueError:
            new_token_sequence.extend(token_sequence[i:])
            break
        new_token_sequence.extend(token_sequence[i:j])
        i = j

        if (
            i + 1 < len(token_sequence)
            and token_sequence[i + 1] == token_pair_[1]
        ):
            indices.add(len(new_token_sequence))
            new_token_sequence.append(token)
            i += 2
        else:
            new_token_sequence.append(token_sequence[i])
            i += 1

    return new_token_sequence, indices


class BPE(SerializeMixin):

    def __init__(
        self,
        codebook_size: int,
        token_pairs: list[TokenPair],
    ) -> None:
        self._codebook_size = codebook_size
        self._token_pairs = token_pairs
        self._encoder = {
            token_pair: i
            for i, token_pair in enumerate(token_pairs, codebook_size)
        }
        self._decoder = ([[i] for i in range(codebook_size)]
                         + [None] * len(token_pairs))

    def __getstate__(self) -> ArgsKwargs:
        args, kwargs = super().__getstate__()
        args = (self._codebook_size, self._token_pairs) + args
        return args, kwargs

    def encode(self, token_sequence: TokenSequence) -> TokenSequence:
        while len(token_sequence) >= 2:
            token_pairs = (  # do not put into try block
                (token, token_pair)
                for token_pair in zip(token_sequence, token_sequence[1:])
                if (token := self._encoder.get(token_pair)) is not None
            )
            try:
                token, token_pair = min(token_pairs)
            except ValueError:
                break
            token_sequence, _ = merge(token_sequence, token_pair, token)
        return token_sequence

    def _decode(self, token: Token) -> TokenSequence:
        if token >= len(self._decoder):
            return [token]
        token_sequence = self._decoder[token]
        if token_sequence is None:
            token_pair = self._token_pairs[token - self._codebook_size]
            token_sequence = (
                self._decode(token_pair[0]) + self._decode(token_pair[1])
            )
            self._decoder[token] = token_sequence
        return token_sequence

    def decode(self, token_sequence: TokenSequence) -> TokenSequence:
        return sum(map(self._decode, token_sequence), [])


def worker(
    connection: Connection,
    token_sequences: list[TokenSequence],
) -> None:
    while True:
        command = connection.recv()
        if command is None:
            connection.send(token_sequences)
            return

        token_pair, token = command

        previous_tokens: Counter[Token] = Counter()
        next_tokens: Counter[Token] = Counter()

        for i, token_sequence in enumerate(token_sequences):
            token_sequence, indices = merge(token_sequence, token_pair, token)
            token_sequences[i] = token_sequence

            for i in indices - {0}:
                previous_tokens[token_sequence[i - 1]] += 1
            for i in indices - {len(token_sequence) - 1}:
                next_tokens[token_sequence[i + 1]] += 1

        connection.send((previous_tokens, next_tokens))


def estimate_compression(
    counter: Counter[TokenPair],
    num_new_tokens: int,
    corpus_size: int,
) -> float:
    most_common = counter.most_common(num_new_tokens)
    return sum(n for _, n in most_common) / corpus_size


class BPETrainer:

    def __init__(
        self,
        *args,
        codebook_size: int,
        max_size: int,
        token_sequences: Iterable[TokenSequence],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._codebook_size = codebook_size
        self._max_size = max_size

        token_sequences = list(token_sequences)

        logger.info("Counting...")
        counter: Counter[TokenPair] = Counter()
        for token_sequence in tqdm.tqdm(token_sequences):
            token_pairs = zip(token_sequence, token_sequence[1:])
            counter.update(token_pairs)
        for token_pair in [
            token_pair for token_pair, n in counter.items() if n <= 1
        ]:
            counter.pop(token_pair)

        estimated_compression = estimate_compression(
            counter,
            max_size - codebook_size,
            sum(map(len, token_sequences)),
        )
        logger.info(
            "Estimated compression: %.2f%%",
            estimated_compression * 100,
        )

        self._counter = counter

        cpu = os.cpu_count()
        assert cpu is not None
        cpu = min(cpu, len(token_sequences))

        connections: list[Connection] = []
        processes: list[Process] = []
        for i in range(cpu):
            connection, child_connection = Pipe()
            process = Process(
                target=worker,
                args=(child_connection, token_sequences[i::cpu]),
            )
            connections.append(connection)
            processes.append(process)
        self._connections = connections
        self._processes = processes

    def start(self) -> None:
        for process in self._processes:
            process.start()

    def join(self) -> None:
        for process in self._processes:
            process.join()

    def _merge(self, token_pair: TokenPair, token: Token) -> None:
        previous_tokens: Counter[Token] = Counter()
        next_tokens: Counter[Token] = Counter()

        for connection in self._connections:
            connection.send((token_pair, token))
        for connection in self._connections:
            previous_tokens_, next_tokens_ = connection.recv()
            previous_tokens.update(previous_tokens_)
            next_tokens.update(next_tokens_)

        for previous_token, n in previous_tokens.items():
            self._counter[previous_token, token] += n
            self._counter[previous_token, token_pair[0]] -= n

        for next_token, n in next_tokens.items():
            self._counter[token, next_token] += n
            self._counter[token_pair[1], next_token] -= n

        for t in range(token):
            if self._counter[t, token_pair[0]] <= 1:
                self._counter.pop((t, token_pair[0]), None)
            if self._counter[token_pair[1], t] <= 1:
                self._counter.pop((token_pair[1], t), None)

        self._counter.pop(token_pair)

    def train(self) -> tuple[BPE, list[TokenSequence]]:
        logger.info("Training started.")

        token_pairs: list[TokenPair] = []
        for token in range(self._codebook_size, self._max_size):
            most_common = self._counter.most_common(1)
            if not most_common:
                break
            (token_pair, n), = most_common
            assert n > 1
            logger.info(
                "Merging (%d, %d) as %d for %d times.",
                *token_pair,
                token,
                n,
            )
            token_pairs.append(token_pair)
            self._merge(token_pair, token)

        logger.info("Training finished.")

        bpe = BPE(self._codebook_size, token_pairs)

        token_sequences: list[TokenSequence] = []
        for connection in self._connections:
            connection.send(None)
            token_sequences_ = connection.recv()
            token_sequences.extend(token_sequences_)

        return bpe, token_sequences
