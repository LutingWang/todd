__all__ = [
    'BaseSegmentor',
]

from typing import Iterable


class BaseSegmentor:

    def __init__(self, *args, max_length: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._max_length = max_length

    def _is_valid(self, text: str) -> bool:
        return self._max_length is None or len(text) <= self._max_length

    def _segment(self, text: str) -> Iterable[str]:
        raise NotImplementedError

    def segment(self, text: str) -> list[str]:
        segments = ['']
        for segment in self._segment(text):
            assert self._is_valid(segment)
            if self._is_valid(segments[-1] + segment):
                segments[-1] += segment
            else:
                segments.append(segment)
        return segments
