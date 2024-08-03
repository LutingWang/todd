__all__ = [
    'SAMed2DDataset',
]

import json
import pathlib
from abc import ABC
from typing import Literal, TypedDict

import torch

from ..registries import DatasetRegistry
from .access_layers import PILAccessLayer
from .pil import PILDataset

Split = Literal['v1']


class T(TypedDict):
    id_: str
    image: torch.Tensor


@DatasetRegistry.register_()
class SAMed2DDataset(PILDataset[T], ABC):
    DATA_ROOT = pathlib.Path('data/sa_med2d')
    ANNOTATIONS_ROOT = DATA_ROOT / 'annotations'
    SUFFIX = 'png'

    def __init__(
        self,
        *args,
        split: Split,
        access_layer: PILAccessLayer | None = None,
        annotations_file: pathlib.Path | str | None = None,
        **kwargs,
    ) -> None:
        if access_layer is None:
            access_layer = PILAccessLayer(
                data_root=str(self.DATA_ROOT),
                task_name='images',
                subfolder_action='none',
                suffix=self.SUFFIX,
            )
        if annotations_file is None:
            annotations_file = self.ANNOTATIONS_ROOT / f'SAMed2D_{split}.json'
        elif isinstance(annotations_file, str):
            annotations_file = pathlib.Path(annotations_file)
        self._annotations_file = annotations_file

        super().__init__(*args, access_layer=access_layer, **kwargs)

    def build_keys(self) -> list[str]:
        with self._annotations_file.open() as f:
            annotations: dict[str, list[str]] = json.load(f)
            return [
                k.removeprefix('images/').removesuffix(f'.{self.SUFFIX}')
                for k in annotations
            ]

    def __getitem__(self, index: int) -> T:
        key, image = self._access(index)
        tensor = self._transform(image)
        return T(id_=key, image=tensor)
