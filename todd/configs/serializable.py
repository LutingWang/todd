__all__ = [
    'SerializableConfig',
]

import difflib
import pathlib
from abc import abstractmethod
from typing_extensions import Self

from .config import Config


class SerializableConfig(Config):

    @classmethod
    @abstractmethod
    def _loads(cls, s: str) -> dict:
        pass

    @classmethod
    def loads(cls, s: str) -> Self:
        return cls(cls._loads(s))  # type: ignore[abstract]

    @classmethod
    def load(cls, file) -> Self:
        file = pathlib.Path(file)
        # `loads` does not support `_delete_` with `_base_`
        config = cls._loads(file.read_text())
        base_config = cls()  # type: ignore[abstract]
        for base in config.pop('_base_', []):
            base_config.update(cls.load(file.parent / base))
        base_config.update(config)
        return base_config

    @abstractmethod
    def dumps(self) -> str:
        pass

    def dump(self, file) -> None:
        r"""Dump the config to a file.

        Args:
            file: the file path.

        Refer to `dumps` for more details.
        """
        pathlib.Path(file).write_text(self.dumps())

    def diff(self, other: Self, html: bool = False) -> str:
        """Diff configs.

        Args:
            other: the other config to diff.
            html: output diff in html format. Default is pure text.

        Returns:
            Diff message.
        """
        a = self.dumps().split('\n')
        b = other.dumps().split('\n')
        if html:
            return difflib.HtmlDiff().make_file(a, b)
        return '\n'.join(difflib.Differ().compare(a, b))
