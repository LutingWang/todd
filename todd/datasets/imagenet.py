__all__ = [
    'ImageNetDataset',
]

import json
import os
import pathlib
from typing import Iterator, Literal, TypedDict

import torch
import torchvision.transforms.functional as F
from PIL import Image

from ..registries import DatasetRegistry
from .access_layers import PILAccessLayer
from .base import BaseDataset

DATA_ROOT = pathlib.Path('data/imagenet')
ANNOTATIONS_ROOT = DATA_ROOT / 'annotations'
SYNSETS_FILE = DATA_ROOT / 'synsets.json'
SUFFIX = 'JPEG'

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


class Keys:

    def __init__(
        self,
        annotations: Annotations,
        synsets: Synsets,
    ) -> None:
        self._annotations = annotations
        self._synsets = synsets

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, index: int) -> str:
        annotation = self._annotations[index]
        return os.path.join(
            self._synsets[annotation['synset_id']]['WNID'],
            annotation['name'].removesuffix(f'.{SUFFIX}'),
        )

    def __iter__(self) -> Iterator[str]:
        for i in range(len(self)):
            yield self[i]


VT = Image.Image


class T(TypedDict):
    id_: str
    image: torch.Tensor
    category: int


@DatasetRegistry.register_()
class ImageNetDataset(BaseDataset[T, str, VT]):

    def __init__(self, *args, split: Split, **kwargs) -> None:
        annotations_file = ANNOTATIONS_ROOT / f'{split}.json'
        with annotations_file.open() as f:
            self._annotations: Annotations = json.load(f)

        with SYNSETS_FILE.open() as f:
            self._synsets: Synsets = {
                int(k): v
                for k, v in json.load(f).items()
            }

        self._categories = {
            synset_id: i
            for i, synset_id in enumerate(sorted(self._synsets))
        }

        access_layer = PILAccessLayer(
            data_root=str(DATA_ROOT),
            task_name=split,
            suffix=SUFFIX,
            subfolder_action='walk',
        )
        super().__init__(*args, access_layer=access_layer, **kwargs)

    def build_keys(self) -> Keys:
        return Keys(self._annotations, self._synsets)

    def __getitem__(self, index: int) -> T:
        key, image = self._access(index)
        tensor = (
            F.pil_to_tensor(image)
            if self._transforms is None else self._transforms(image)
        )
        synset_id = self._annotations[index]['synset_id']
        category = self._categories[synset_id]
        return T(id_=key, image=tensor, category=category)
