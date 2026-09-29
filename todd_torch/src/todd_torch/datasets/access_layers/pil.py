__all__ = [
    'PILAccessLayer',
]

from PIL import Image

from ..registries import AccessLayerRegistry
from .directory import DirectoryAccessLayer
from .suffix import SuffixMixin

V = Image.Image


@AccessLayerRegistry.register_()
class PILAccessLayer(SuffixMixin[V], DirectoryAccessLayer[V]):

    def __getitem__(self, key: str) -> V:
        return Image.open(self._file(key))

    def __setitem__(self, key: str, value: V) -> None:
        value.save(self._file(key))
