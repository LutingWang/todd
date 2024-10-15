__all__ = [
    'V3DetDataset',
]

import pathlib
from typing import Literal

from ..registries import DatasetRegistry
from .access_layers.pil import PILAccessLayer
from .coco import COCODataset

Split = Literal['train', 'val']
Year = Literal[2023]
Version = Literal['v1']


@DatasetRegistry.register_()
class V3DetDataset(COCODataset):
    DATA_ROOT = pathlib.Path('data/v3det')
    ANNOTATIONS_ROOT = DATA_ROOT / 'annotations'

    def __init__(
        self,
        *args,
        split: Split,
        year: Year = 2023,
        version: Version = 'v1',
        access_layer: PILAccessLayer | None = None,
        annotations_file: pathlib.Path | str | None = None,
        **kwargs,
    ) -> None:
        if access_layer is None:
            access_layer = PILAccessLayer(
                data_root=str(self.DATA_ROOT),
                suffix='jpg',
            )
        if annotations_file is None:
            annotations_file = (
                self.ANNOTATIONS_ROOT / f'v3det_{year}_{version}_{split}.json'
            )

        super().__init__(
            *args,
            split=split,
            # year=...,  # not used
            access_layer=access_layer,
            annotations_file=annotations_file,
            **kwargs,
        )
