from typing import Union

from mmcv.utils import Registry

from .base import BaseDistiller


DISTILLERS = Registry('distillers')

DistillerConfig = Union[BaseDistiller, dict]


def build_distiller(cfg: DistillerConfig) -> BaseDistiller:
    if isinstance(cfg, BaseDistiller):
        return cfg
    assert isinstance(cfg, dict)
    return DISTILLERS.build(cfg)
