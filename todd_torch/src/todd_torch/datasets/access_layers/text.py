__all__ = [
    'TextAccessLayer',
]

from ..registries import AccessLayerRegistry
from .folder import FolderAccessLayer
from .suffix import SuffixMixin


@AccessLayerRegistry.register_()
class TextAccessLayer(SuffixMixin[str], FolderAccessLayer[str]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, suffix='txt', **kwargs)

    def __getitem__(self, key: str) -> str:
        return self._file(key).read_text()

    def __setitem__(self, key: str, value: str) -> None:
        self._file(key).write_text(value)
