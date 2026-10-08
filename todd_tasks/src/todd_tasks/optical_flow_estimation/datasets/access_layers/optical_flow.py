__all__ = [
    'OpticalFlowAccessLayer',
]

from typing import TypeVar

from todd import Config
from todd.registries import BuildPreHookMixin, Item, RegistryMeta
from todd_torch.datasets.access_layers import FileAccessLayer, SuffixMixin

from ...optical_flow import SerializeMixin
from ...registries import OFEOpticalFlowRegistry
from ..registries import OFEAccessLayerRegistry

V = TypeVar('V', bound=SerializeMixin)


@OFEAccessLayerRegistry.register_()
class OpticalFlowAccessLayer(
    BuildPreHookMixin,
    SuffixMixin[V],
    FileAccessLayer[V],
):

    def __init__(self, *args, optical_flow_type: type[V], **kwargs) -> None:
        super().__init__(
            *args,
            suffix=optical_flow_type.SUFFIX,
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

    def __getitem__(self, key: str) -> V:
        return self._optical_flow_type.load(self._path(key))

    def __setitem__(self, key: str, value: V) -> None:
        value.dump(self._path(key))
