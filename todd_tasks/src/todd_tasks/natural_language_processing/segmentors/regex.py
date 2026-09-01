__all__ = [
    'RegexSegmentor',
]

import re
from typing import Iterable

from .base import BaseSegmentor


class RegexSegmentor(BaseSegmentor):

    def __init__(self, *args, regex: re.Pattern[str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._regex = regex

    def _segment(self, text: str) -> Iterable[str]:
        return self._regex.split(text)
