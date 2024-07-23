import pytest

from todd import Config
from todd.runners.strategies import BaseStrategy
from todd.tasks.natural_language_processing import NLPRunnerRegistry
from todd.tasks.natural_language_processing.bpe import (
    BPE,
    BPECallback,
    BPETrainer,
    merge,
)
from todd.tasks.natural_language_processing.runners import NLPCallbackRegistry


def test_merge() -> None:
    assert merge([1, 2, 3, 2, 3, 4], (2, 3), 5) == ([1, 5, 5, 4], {1, 2})
    assert merge([1, 2, 3, 4, 5], (4, 5), 6) == ([1, 2, 3, 6], {3})
    assert merge([], (1, 2), 3) == ([], set())
    assert merge([1, 2, 3, 4, 5, 2, 3], (2, 3), 6) == ([1, 6, 4, 5, 6], {1, 4})
    assert merge([1, 2, 2, 2, 3], (2, 2), 6) == ([1, 6, 2, 3], {1})
    assert merge([1, 2, 2, 3, 2, 3], (2, 3), 6) == ([1, 2, 6, 6], {2, 3})


class TestBPE:

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self._bpe = BPE(5, [(1, 2), (0, 1)])

    def test_encode(self) -> None:
        assert self._bpe.encode([0, 1, 2, 1, 0]) == [0, 5, 1, 0]

    def test__decode(self) -> None:
        assert self._bpe._decode(5) == [1, 2]

    def test_decode(self) -> None:
        assert self._bpe.decode([5, 6]) == [1, 2, 0, 1]


class TestBPETrainer:

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        config = Config(
            type=BPETrainer.__name__,
            name='bpe',
            strategy=dict(type=BaseStrategy.__name__),
            callbacks=[
                dict(
                    type=(
                        f'{NLPCallbackRegistry.__name__}.'
                        f'{BPECallback.__name__}'
                    ),
                ),
            ],
            token_sequences=[
                [0, 1, 2, 3, 4],
                [0, 1, 1, 3, 4],
                [4, 3, 2, 1, 0],
            ],
            codebook_size=5,
            max_size=10,
        )

        self._bpe_trainer: BPETrainer = NLPRunnerRegistry.build(config)

    def test__run_iter(self) -> None:
        self._bpe_trainer._run_iter(5, dict(token_pair=(1, 2)))
        assert self._bpe_trainer._token_sequences == [
            [0, 5, 3, 4],
            [0, 1, 1, 3, 4],
            [4, 3, 2, 1, 0],
        ]

    def test_run(self) -> None:
        memo = self._bpe_trainer.run()
        bpe = memo['bpe']
        assert isinstance(bpe, BPE)
        assert bpe._codebook_size == self._bpe_trainer._codebook_size
        assert bpe._token_pairs == [(0, 1), (3, 4)]
