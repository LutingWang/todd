__all__ = [
    'PILAccessLayer',
]

from PIL import Image

from ..registries import AccessLayerRegistry
from .file import FileAccessLayer
from .suffix import SuffixMixin

V = Image.Image


@AccessLayerRegistry.register_()
class PILAccessLayer(SuffixMixin[V], FileAccessLayer[V]):

    def __getitem__(self, key: str) -> V:
        return Image.open(self._path(key))

    def __setitem__(self, key: str, value: V) -> None:
        value.save(self._path(key))
