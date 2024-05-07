__all__ = [
    'DatasetRegistry',
]

import inspect
from functools import partial
from typing import cast

import torch.utils.data.dataset

from .registry import BuildSpec, Item, Registry


class DatasetRegistry(Registry):
    pass


for _, c in inspect.getmembers(torch.utils.data.dataset, inspect.isclass):
    if issubclass(c, torch.utils.data.Dataset):
        DatasetRegistry.register_()(cast(Item, c))

DatasetRegistry.register_(
    force=True,
    build_spec=BuildSpec(datasets=partial(map, DatasetRegistry.build)),
)(torch.utils.data.ConcatDataset)
