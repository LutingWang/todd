__all__ = [
    'ema',
    'EMA',
]

from typing import Any


def ema(x: Any, y: Any, decay: Any) -> Any:
    return x * decay + y * (1 - decay)


class EMA:

    def __init__(self, decay: Any = 0.99) -> None:
        self._decay = decay

    @classmethod
    def check_decay(cls, decay: Any) -> None:
        assert 0 <= decay <= 1

    @property
    def decay(self) -> Any:
        return self._decay

    def __call__(self, x: Any, y: Any) -> Any:
        if x is None:
            assert y is not None
            return y
        if y is None:
            return x
        return ema(x, y, self._decay)
