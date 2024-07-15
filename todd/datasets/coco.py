__all__ = [
    'COCODataset',
]

import pathlib
from typing import Iterator, Literal, TypedDict

import torch
from pycocotools.coco import COCO

from ..registries import DatasetRegistry
from .access_layers import PILAccessLayer
from .pil import PILDataset

Split = Literal['train', 'val']


class Keys:

    def __init__(self, coco: COCO, suffix: str) -> None:
        self._coco = coco
        self._suffix = suffix
        self._image_ids = coco.getImgIds()

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, index: int) -> str:
        image_id = self._image_ids[index]
        image, = self._coco.loadImgs(image_id)
        return image['file_name'].removesuffix(f'.{self._suffix}')

    def __iter__(self) -> Iterator[str]:
        for i in range(len(self)):
            yield self[i]


class T(TypedDict):  # pylint: disable=duplicate-code
    id_: str
    image: torch.Tensor


@DatasetRegistry.register_()
class COCODataset(PILDataset[T]):
    DATA_ROOT = pathlib.Path('data/coco')
    ANNOTATIONS_ROOT = DATA_ROOT / 'annotations'
    YEAR = 2017
    SUFFIX = 'jpg'

    def __init__(self, *args, split: Split, **kwargs) -> None:
        split_year = f'{split}{self.YEAR}'

        annotations_file = (
            self.ANNOTATIONS_ROOT / f'instances_{split_year}.json'
        )
        self._coco = COCO(annotations_file)

        access_layer = PILAccessLayer(
            data_root=str(self.DATA_ROOT),
            task_name=split_year,
            suffix=self.SUFFIX,
        )
        super().__init__(*args, access_layer=access_layer, **kwargs)

    def build_keys(self) -> Keys:
        return Keys(self._coco, self.SUFFIX)

    def __getitem__(self, index: int) -> T:
        key, image = self._access(index)
        tensor = self._transform(image)
        return T(id_=key, image=tensor)
