__all__ = [
    'OpticalFlowAccessLayer',
]

from typing import TypeVar

from todd.datasets.access_layers import FolderAccessLayer, SuffixMixin
from todd.patches.py import classproperty
from todd.registries import BuildSpec, BuildSpecMixin

from ...optical_flow import SerializeMixin
from ...registries import OFEOpticalFlowRegistry
from ..registries import OFEAccessLayerRegistry

VT = TypeVar('VT', bound=SerializeMixin)


@OFEAccessLayerRegistry.register_()
class OpticalFlowAccessLayer(
    BuildSpecMixin,
    SuffixMixin[VT],
    FolderAccessLayer[VT],
):

    def __init__(self, *args, optical_flow_type: type[VT], **kwargs) -> None:
        super().__init__(
            *args,
            suffix=optical_flow_type.SUFFIX.removeprefix('.'),
            **kwargs,
        )
        self._optical_flow_type = optical_flow_type

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec = BuildSpec(
            optical_flow_type=lambda c: OFEOpticalFlowRegistry[c.type],
        )
        return super().build_spec | build_spec

    def __getitem__(self, key: str) -> VT:
        return self._optical_flow_type.load(self._file(key))

    def __setitem__(self, key: str, value: VT) -> None:
        value.dump(self._file(key))
