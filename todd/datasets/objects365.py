# pylint: disable=duplicate-code

__all__ = [
    'Objects365Dataset',
]

import pathlib
from collections import UserList
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Literal,
    Mapping,
    TypedDict,
    cast,
)
from typing_extensions import Self

import torch
from pycocotools.coco import COCO

from ..registries import DatasetRegistry
from .access_layers import PILAccessLayer
from .coco import BaseDataset as BaseCOCODataset
from .coco import Keys as COCOKeys

if TYPE_CHECKING:
    from todd.tasks.object_detection import BBox, FlattenBBoxesXYWH

Split = Literal['train', 'val']


class Keys(COCOKeys):

    def __init__(self, *args, ignore: Iterable[str], **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if ignore := set(ignore):
            ignore_indices = [i for i in range(len(self)) if self[i] in ignore]
            assert len(ignore_indices) == len(ignore)
            for i in ignore_indices[::-1]:
                self._image_ids.pop(i)

    def _getitem(self, image_id: int) -> str:
        item = super()._getitem(image_id)
        return item.removeprefix('images/')


@dataclass(frozen=True)
class Annotation:
    area: float
    bbox: 'BBox'
    category: int
    is_crowd: bool
    is_fake: bool
    is_reflected: bool

    @classmethod
    def load(
        cls,
        annotation: Mapping[str, Any],
        categories: Mapping[int, int],
    ) -> Self:
        return cls(
            annotation['area'],
            cast('BBox', annotation['bbox']),
            categories[annotation['category_id']],
            bool(annotation['iscrowd']),
            bool(annotation['isfake']),
            bool(annotation['isreflected']),
        )


class Annotations(UserList[Annotation]):

    @classmethod
    def load(
        cls,
        coco: COCO,
        image_id: int,
        categories: Mapping[int, int],
    ) -> Self:
        annotation_ids = coco.getAnnIds([image_id])
        annotations = coco.loadAnns(annotation_ids)
        return cls(
            Annotation.load(annotation, categories)
            for annotation in annotations
        )

    @property
    def areas(self) -> torch.Tensor:
        return torch.tensor([annotation.area for annotation in self])

    @property
    def bboxes(self) -> 'FlattenBBoxesXYWH':
        from todd.tasks.object_detection import FlattenBBoxesXYWH
        if self:
            bboxes = torch.tensor([annotation.bbox for annotation in self])
        else:
            bboxes = torch.zeros(0, 4)
        return FlattenBBoxesXYWH(bboxes)

    @property
    def categories(self) -> torch.Tensor:
        return torch.tensor([annotation.category for annotation in self])

    @property
    def is_crowd(self) -> torch.Tensor:
        return torch.tensor([annotation.is_crowd for annotation in self])

    @property
    def is_fake(self) -> torch.Tensor:
        return torch.tensor([annotation.is_fake for annotation in self])

    @property
    def is_reflected(self) -> torch.Tensor:
        return torch.tensor([annotation.is_reflected for annotation in self])


class T(TypedDict):
    id_: str
    image: torch.Tensor
    annotations: Annotations


@DatasetRegistry.register_()
class Objects365Dataset(BaseCOCODataset[COCO, T]):
    _keys: Keys

    DATA_ROOT = pathlib.Path('data/objects365')
    ANNOTATIONS_ROOT = DATA_ROOT / 'annotations'

    IGNORE_KEYS = (
        'v1/patch6/objects365_v1_00320532',
        'v1/patch6/objects365_v1_00320534',
        'v2/patch16/objects365_v2_00908726',
    )

    def __init__(
        self,
        *args,
        split: Split,
        version: Literal['v1', 'v2'] = 'v2',
        access_layer: PILAccessLayer | None = None,
        annotations_file: pathlib.Path | str | None = None,
        **kwargs,
    ) -> None:
        self._split = split

        if access_layer is None:
            access_layer = PILAccessLayer(
                data_root=str(self.DATA_ROOT),
                task_name=split,
                suffix=self.SUFFIX,
            )
        if annotations_file is None:
            annotations_file = (
                self.ANNOTATIONS_ROOT / f'zhiyuan_obj{version}_{split}.json'
            )

        coco = COCO(annotations_file)

        self._categories = {
            category_id: i
            for i, category_id in enumerate(coco.getCatIds())
        }

        super().__init__(*args, api=coco, access_layer=access_layer, **kwargs)

    def build_keys(self) -> Keys:
        ignore_keys: Iterable[str]
        if self._split == 'train':
            ignore_keys = self.IGNORE_KEYS
        elif self._split == 'val':
            ignore_keys = tuple()
        else:
            raise ValueError(f"Unknown split: {self._split}")
        return Keys(self._api, self.SUFFIX, ignore=ignore_keys)

    def __getitem__(self, index: int) -> T:
        key, image = self._access(index)
        tensor = self._transform(image)
        annotations = (
            Annotations.load(
                self._api,
                self._keys.image_ids[index],
                self._categories,
            ) if self._load_annotations else Annotations()
        )
        return T(id_=key, image=tensor, annotations=annotations)
