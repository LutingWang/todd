from typing import Union

from mmcv.utils import Registry

from .base import BaseDataset


DATASETS = Registry('datasets')

DatasetConfig = Union[BaseDataset, dict]


def build_dataset(cfg: DatasetConfig) -> BaseDataset:
    if isinstance(cfg, BaseDataset):
        return cfg
    assert isinstance(cfg, dict)
    return DATASETS.build(cfg)
