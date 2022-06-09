from typing import Union

from ..utils import build_metas
from .base import BaseAccessLayer, BaseDataset

ACCESS_LAYERS, build_access_layer = build_metas(
    'access_layers', BaseAccessLayer,  # type: ignore[misc]
)
AccessLayerConfig = Union[BaseAccessLayer, dict]

DATASETS, build_dataset = build_metas('datasets', BaseDataset)
DatasetConfig = Union[BaseDataset, dict]
