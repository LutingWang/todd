import pytest

from todd.tasks.natural_language_processing.bpe import BPE, Trainer, merge


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


class TestTrainer:

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        token_sequences = [
            [0, 1, 2, 3, 4],
            [0, 1, 1, 3, 4],
            [4, 3, 2, 1, 0],
        ]
        self._trainer = Trainer(token_sequences, 5, 10)

    def test_merge(self) -> None:
        self._trainer.merge((1, 2), 5)
        assert self._trainer._token_sequences == [
            [0, 5, 3, 4],
            [0, 1, 1, 3, 4],
            [4, 3, 2, 1, 0],
        ]

    def test_train(self) -> None:
        bpe = self._trainer.train()
        assert isinstance(bpe, BPE)
        assert bpe._codebook_size == self._trainer._codebook_size
        assert bpe._token_pairs == [(0, 1), (3, 4)]
