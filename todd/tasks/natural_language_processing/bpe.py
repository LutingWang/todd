__all__ = [
    'BPE',
    'BPETrainer',
    'BPECallback',
]

from collections import Counter
from typing import Any, Iterable, Mapping, TypeVar

import torch
from torch import nn

from todd import Config, RegistryMeta
from todd.bases.registries import Item
from todd.runners import BaseRunner, Memo
from todd.runners.callbacks import BaseCallback
from todd.utils import ArgsKwargs, SerializeMixin

from .registries import NLPRunnerRegistry
from .runners import NLPCallbackRegistry

Token = int
TokenPair = tuple[Token, Token]
TokenSequence = list[Token]

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
        if token_sequence[i:i + 2] == token_pair_:
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
        self._token_mappings = ([[i] for i in range(codebook_size)]
                                + [None] * len(token_pairs))

    def __getstate__(self) -> ArgsKwargs:
        args, kwargs = super().__getstate__()
        args = (self._codebook_size, self._token_pairs) + args
        return args, kwargs

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


@NLPRunnerRegistry.register_()
class BPETrainer(BaseRunner[T]):

    def __init__(
        self,
        *args,
        token_sequences: Iterable[TokenSequence],
        codebook_size: int,
        max_size: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._token_sequences = list(token_sequences)
        self._codebook_size = codebook_size
        self._max_size = max_size
        self._counter = self._count()

    def _count(self) -> Counter[TokenPair]:
        counter: Counter[TokenPair] = Counter()
        for token_sequence in self._token_sequences:
            token_pairs = zip(token_sequence, token_sequence[1:])
            counter.update(token_pairs)
        return counter

    @property
    def token_sequences(self) -> list[TokenSequence]:
        return self._token_sequences

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def counter(self) -> Counter[TokenPair]:
        return self._counter

    def _init_model(self, *args, **kwargs) -> None:
        pass

    @property
    def iters(self) -> int:
        return self._max_size - self._codebook_size

    @classmethod
    def model_build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        return config

    @classmethod
    def dataset_build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config.dataset = None
        return config

    @classmethod
    def dataloader_build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config.dataloader = None
        return config

    def _run_iter(self, batch: Token, memo: Memo, *args, **kwargs) -> Memo:
        token = batch
        token_pair: TokenPair = memo['token_pair']

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

        return memo

    def _setup(self) -> Memo:
        memo = super()._setup()
        memo['dataloader'] = range(self._codebook_size, self._max_size)
        return memo

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *args,
        **kwargs,
    ) -> None:
        super().load_state_dict(state_dict, *args, **kwargs)
        self._token_sequences = state_dict['token_sequences']
        self._counter = self._count()

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['token_sequences'] = self._token_sequences
        return state_dict


@NLPCallbackRegistry.register_()
class BPECallback(BaseCallback[T]):
    runner: BPETrainer

    def before_run(self, memo: Memo) -> None:
        super().before_run(memo)
        memo['token_pairs'] = []

    def should_break(self, batch: Token, memo: Memo) -> bool:
        if super().should_break(batch, memo):
            return True
        most_common = self.runner.counter.most_common(1)
        if not most_common:
            return True
        (token_pair, n), = most_common
        if n <= 1:
            return True
        memo.update(token_pair=token_pair)

        log: dict[str, Any] | None = memo.get('log')
        if log is not None:
            log.update(token_pair=token_pair, token=batch, n=n)

        return False

    def after_run_iter(self, batch: Token, memo: Memo) -> None:
        super().after_run_iter(batch, memo)
        token_pairs: list[TokenPair] = memo['token_pairs']
        token_pair: TokenPair = memo['token_pair']
        token_pairs.append(token_pair)
        self.runner.counter.pop(token_pair)

    def after_run(self, memo: Memo) -> None:
        super().after_run(memo)
        bpe = BPE(self.runner._codebook_size, memo['token_pairs'])
        memo['bpe'] = bpe

        torch.save(bpe, self.runner.work_dir / 'bpe.pth')
        torch.save(
            self.runner.token_sequences,
            self.runner.work_dir / 'token_sequences.pth',
        )
