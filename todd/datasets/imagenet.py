__all__ = [
    'ImageNetDataset',
]

import json
import os
import pathlib
from abc import ABC
from typing import Literal, TypedDict

import torch

from ..registries import DatasetRegistry
from .access_layers import PILAccessLayer
from .base import KeysProtocol
from .pil import PILDataset

Split = Literal['train', 'val']


class Synset(TypedDict):
    WNID: str
    words: str
    gloss: str
    num_children: int
    children: list[int]
    wordnet_height: int
    num_train_images: int


Synsets = dict[int, Synset]


class Annotation(TypedDict):
    name: str
    synset_id: int


Annotations = list[Annotation]


class Keys(KeysProtocol[str]):  # pylint: disable=unsubscriptable-object

    def __init__(
        self,
        annotations: Annotations,
        synsets: Synsets,
        suffix: str,
    ) -> None:
        self._annotations = annotations
        self._synsets = synsets
        self._suffix = suffix

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, index: int) -> str:
        annotation = self._annotations[index]
        return os.path.join(
            self._synsets[annotation['synset_id']]['WNID'],
            annotation['name'].removesuffix(f'.{self._suffix}'),
        )


class T(TypedDict):
    id_: str
    image: torch.Tensor
    category: int


@DatasetRegistry.register_()
class ImageNetDataset(PILDataset[T], ABC):
    DATA_ROOT = pathlib.Path('data/imagenet')
    ANNOTATIONS_ROOT = DATA_ROOT / 'annotations'
    SYNSETS_FILE = DATA_ROOT / 'synsets.json'
    SUFFIX = 'JPEG'

    def __init__(
        self,
        *args,
        split: Split,
        access_layer: PILAccessLayer | None = None,
        annotations_file: pathlib.Path | str | None = None,
        synsets_file: pathlib.Path | str | None = None,
        **kwargs,
    ) -> None:
        if access_layer is None:
            access_layer = PILAccessLayer(
                data_root=str(self.DATA_ROOT),
                task_name=split,
                subfolder_action='walk',
                suffix=self.SUFFIX,
            )
        if annotations_file is None:
            annotations_file = self.ANNOTATIONS_ROOT / f'{split}.json'
        elif isinstance(annotations_file, str):
            annotations_file = pathlib.Path(annotations_file)
        if synsets_file is None:
            synsets_file = self.SYNSETS_FILE
        elif isinstance(synsets_file, str):
            synsets_file = pathlib.Path(synsets_file)

        with annotations_file.open() as f:
            self._annotations: Annotations = json.load(f)

        with self.SYNSETS_FILE.open() as f:
            synsets: dict[str, Synset] = json.load(f)
        synsets_: Synsets = {int(k): v for k, v in synsets.items()}
        self._synsets = synsets_

        self._categories = {
            synset_id: i
            for i, synset_id in enumerate(sorted(synsets_))
        }

        super().__init__(*args, access_layer=access_layer, **kwargs)

    def build_keys(self) -> Keys:
        return Keys(self._annotations, self._synsets, self.SUFFIX)

    def __getitem__(self, index: int) -> T:
        key, image = self._access(index)
        tensor = self._transform(image)
        synset_id = self._annotations[index]['synset_id']
        category = self._categories[synset_id]
        return T(id_=key, image=tensor, category=category)
