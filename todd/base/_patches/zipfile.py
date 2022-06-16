import functools
import os
import sys
import typing
import zipfile

from .._extensions import get_logger

if sys.version_info < (3, 8):
    get_logger().warning("Monkey patching `zipfile.Path`.")

    class Path:

        def __init__(self, root: str, at: str = ''):
            self._root = zipfile.ZipFile(root)
            self._at = at

        def _next(self, at: str) -> 'Path':
            return Path(self._root.filename, at)

        def _is_child(self, path: 'Path') -> bool:
            return (
                os.path.dirname(path._at.rstrip("/")) == self._at.rstrip("/")
            )

        @functools.cached_property
        def _namelist(self) -> typing.List[str]:
            return self._root.namelist()

        def read_bytes(self) -> bytes:
            return self._root.read(self._at)

        def exists(self) -> bool:
            return self._at in self._namelist

        def iterdir(self) -> typing.Iterator['Path']:
            subs = map(self._next, self._namelist)
            return filter(self._is_child, subs)

        def __str__(self):
            return os.path.join(self._root.filename, self._at)

        def __truediv__(self, at: str) -> 'Path':
            return self._next(os.path.join(self._at, at))

    zipfile.Path = Path
