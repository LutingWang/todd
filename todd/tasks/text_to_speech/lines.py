__all__ = [
    'Lines',
]

import pathlib
import re
from typing import Generator, Mapping

from todd.patches.py_ import remove_prefix

from .utils import normalize_text
from .voices import Voice


class Lines:
    VOICE_PATTERN = re.compile(r'^\[(\w+)\]\s*')

    def __init__(self, *args, voices: Mapping[str, Voice], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._voices = voices

    def parse(
        self,
        file: pathlib.Path,
    ) -> Generator[tuple[Voice, str], None, None]:
        with file.open() as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue

                match_ = self.VOICE_PATTERN.match(line)
                if match_ is None:
                    voice = 'default'
                else:
                    voice = match_[1]
                    line = remove_prefix(line, match_[0])
                yield self._voices[voice], normalize_text(line)
