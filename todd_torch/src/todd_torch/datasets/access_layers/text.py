__all__ = [
    'TextAccessLayer',
]

from ..registries import AccessLayerRegistry
from .file import FileAccessLayer
from .suffix import SuffixMixin


@AccessLayerRegistry.register_()
class TextAccessLayer(SuffixMixin[str], FileAccessLayer[str]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, suffix='.txt', **kwargs)

    def __getitem__(self, key: str) -> str:
        return self._path(key).read_text()

    def __setitem__(self, key: str, value: str) -> None:
        self._path(key).write_text(value)
