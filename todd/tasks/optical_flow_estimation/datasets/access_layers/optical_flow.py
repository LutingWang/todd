__all__ = [
    'OpticalFlowAccessLayer',
]

from typing import TypeVar

from todd import Config, RegistryMeta
from todd.bases.registries import BuildPreHookMixin, Item
from todd.datasets.access_layers import FolderAccessLayer, SuffixMixin

from ...optical_flow import SerializeMixin
from ...registries import OFEOpticalFlowRegistry
from ..registries import OFEAccessLayerRegistry

VT = TypeVar('VT', bound=SerializeMixin)


@OFEAccessLayerRegistry.register_()
class OpticalFlowAccessLayer(
    BuildPreHookMixin,
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

    @classmethod
    def build_pre_hook(
        cls,
        config: Config,
        registry: RegistryMeta,
        item: Item,
    ) -> Config:
        config = super().build_pre_hook(config, registry, item)
        optical_flow_type = config.optical_flow_type
        if isinstance(optical_flow_type, str):
            config.optical_flow_type = (
                OFEOpticalFlowRegistry[optical_flow_type]
            )
        return config

    def __getitem__(self, key: str) -> VT:
        return self._optical_flow_type.load(self._file(key))

    def __setitem__(self, key: str, value: VT) -> None:
        value.dump(self._file(key))
