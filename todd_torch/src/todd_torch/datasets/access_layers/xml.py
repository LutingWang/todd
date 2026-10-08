__all__ = [
    'XMLAccessLayer',
]

from xml.etree import ElementTree  # nosec B405

from defusedxml.ElementTree import parse

from ..registries import AccessLayerRegistry
from .file import FileAccessLayer
from .suffix import SuffixMixin


@AccessLayerRegistry.register_()
class XMLAccessLayer(
    SuffixMixin[ElementTree.ElementTree],
    FileAccessLayer[ElementTree.ElementTree],
):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, suffix='.xml', **kwargs)

    def __getitem__(self, key: str) -> ElementTree.ElementTree:
        return parse(self._path(key))

    def __setitem__(
        self,
        key: str,
        value: ElementTree.ElementTree,
    ) -> None:
        value.write(self._path(key))
