__all__ = [
    'FloAccessLayer',
]

from .....datasets.access_layers import FolderAccessLayer, SuffixMixin
from ...optical_flow import FloOpticalFlow
from ..registries import OFEAccessLayerRegistry

VT = FloOpticalFlow


@OFEAccessLayerRegistry.register_()
class FloAccessLayer(SuffixMixin[VT], FolderAccessLayer[VT]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            suffix='flo',
            subfolder_action='walk',
            **kwargs,
        )

    def __getitem__(self, key: str) -> VT:
        return FloOpticalFlow.load(self._file(key))

    def __setitem__(self, key: str, value: VT) -> None:
        value.dump(self._file(key))
