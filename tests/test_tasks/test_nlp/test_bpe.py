import pytest

from todd.tasks.natural_language_processing.bpe import BPE, merge


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
