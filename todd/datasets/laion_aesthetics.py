__all__ = [
    'LAIONAestheticsDataset',
]

import csv
import pathlib
from abc import ABC
from typing import Literal, TypedDict

import torch

from ..registries import DatasetRegistry
from .access_layers import PILAccessLayer
from .base import KeysProtocol
from .pil import PILDataset

Split = Literal['v2_6.5plus']


class Annotation(TypedDict):
    filename: str
    caption: str
    score: float
    url: str


Annotations = list[Annotation]


class Keys(KeysProtocol[str]):  # pylint: disable=unsubscriptable-object

    def __init__(self, annotations: Annotations) -> None:
        self._annotations = annotations

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, index: int) -> str:
        return self._annotations[index]['filename']


class T(TypedDict):
    id_: str
    image: torch.Tensor
    caption: str
    score: float


@DatasetRegistry.register_()
class LAIONAestheticsDataset(PILDataset[T], ABC):
    DATA_ROOT = pathlib.Path('data/laion/aesthetics')
    ANNOTATIONS_ROOT = DATA_ROOT / 'annotations'
    SUFFIX = None

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
                task_name=split,
                subfolder_action='none',
                suffix=self.SUFFIX,
            )
        if annotations_file is None:
            annotations_file = self.ANNOTATIONS_ROOT / f'{split}.tsv'
        elif isinstance(annotations_file, str):
            annotations_file = pathlib.Path(annotations_file)

        with annotations_file.open() as f:
            self._annotations = [
                Annotation(
                    filename=annotation[0],
                    caption='\t'.join(annotation[1:-2]),
                    score=float(annotation[-2]),
                    url=annotation[-1],
                ) for annotation in csv.reader(f, delimiter='\t')
            ]

        super().__init__(*args, access_layer=access_layer, **kwargs)

    def build_keys(self) -> Keys:
        return Keys(self._annotations)

    def __getitem__(self, index: int) -> T:
        key, image = self._access(index)
        tensor = self._transform(image)
        annotation = self._annotations[index]
        return T(
            id_=key,
            image=tensor,
            caption=annotation['caption'],
            score=annotation['score'],
        )
