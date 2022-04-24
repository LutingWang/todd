from ..utils import build_metas

from .base import BaseAccessLayer, BaseDataset


ACCESS_LAYERS, AccessLayerConfig, build_access_layer = build_metas('access_layers', BaseAccessLayer)
DATASETS, DatasetConfig, build_dataset = build_metas('datasets', BaseDataset)
