__all__ = [
    'ViTRegistry',
    'ConvNeXtRegistry',
]

from torchvision import models

from ..bases.configs import Config
from ..bases.registries import Item, RegistryMeta
from ..patches.py_ import get_
from .registries import TorchVisionRegistry


def build_pre_hook(
    config: Config,
    registry: RegistryMeta,
    item: Item,
) -> Config:
    config.weights = get_(models, config.weights)
    return config


class ViTRegistry(TorchVisionRegistry):
    pass


register_vit = ViTRegistry.register_(build_pre_hook=build_pre_hook)
register_vit(models.VisionTransformer)
register_vit(models.vit_b_16)
register_vit(models.vit_b_32)
register_vit(models.vit_l_16)
register_vit(models.vit_l_32)
register_vit(models.vit_h_14)


class ConvNeXtRegistry(TorchVisionRegistry):
    pass


register_convnext = ConvNeXtRegistry.register_(build_pre_hook=build_pre_hook)
register_convnext(models.ConvNeXt)
register_convnext(models.convnext_tiny)
register_convnext(models.convnext_small)
register_convnext(models.convnext_base)
register_convnext(models.convnext_large)
