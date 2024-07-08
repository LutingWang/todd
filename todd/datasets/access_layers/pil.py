__all__ = [
    'PILAccessLayer',
]

from PIL import Image

from ..registries import AccessLayerRegistry
from .folder import FolderAccessLayer
from .suffix import SuffixMixin

VT = Image.Image


@AccessLayerRegistry.register_()
class PILAccessLayer(SuffixMixin[VT], FolderAccessLayer[VT]):

    def __getitem__(self, key: str) -> VT:
        return Image.open(self._file(key))

    def __setitem__(self, key: str, value: VT) -> None:
        value.save(self._file(key))
