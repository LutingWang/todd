__all__ = [
    'SerializeMixin',
]

import difflib
import pathlib
from abc import abstractmethod
from typing import Any
from typing_extensions import Self

from ..bases.configs import Config
from ..loggers import master_logger


class SerializeMixin(Config):

    @classmethod
    @abstractmethod
    def _loads(cls, __s: str, **kwargs) -> dict[str, Any]:
        pass

    @classmethod
    def loads(cls, s: str, **kwargs) -> Self:
        return cls(cls._loads(s, **kwargs))  # type: ignore[abstract]

    @classmethod
    def load(cls, file: str | pathlib.Path, **kwargs) -> Self:
        if kwargs:
            kwargs_str = ', '.join(f'{k}={v}' for k, v in kwargs.items())
            master_logger.debug(
                "Loading config from %s with %s",
                file,
                kwargs_str,
            )
        else:
            master_logger.debug("Loading config from %s", file)

        if isinstance(file, str):
            file = pathlib.Path(file)
        # do not use `loads`, since it does not support `_delete_` with
        # `_base_`
        config = cls._loads(file.read_text(), **kwargs)
        base_config = cls()  # type: ignore[abstract]
        for base in config.pop('_base_', []):
            if isinstance(base, str):
                base = cls.load(file.parent / base, **kwargs)
            base_config.update(base)
        # do not use __ior__, which does not support recursive update
        base_config.update(config.get('_export_', config))
        return base_config

    @abstractmethod
    def dumps(self) -> str:
        pass

    def dump(self, file: str | pathlib.Path) -> None:
        r"""Dump the config to a file.

        Args:
            file: the file path.

        Refer to `dumps` for more details.
        """
        if isinstance(file, str):
            file = pathlib.Path(file)
        file.write_text(self.dumps())

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
