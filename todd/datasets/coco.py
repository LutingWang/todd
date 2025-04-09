__all__ = [
    'coco_url',
    'COCODataset',
]

import pathlib
from abc import ABC, abstractmethod
from collections import UserList
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Generic,
    Iterable,
    Literal,
    Mapping,
    TypedDict,
    TypeVar,
    cast,
)
from typing_extensions import Self

import torch
from pycocotools.coco import COCO

from ..registries import DatasetRegistry
from .access_layers import PILAccessLayer
from .base import KeysProtocol
from .pil import PILDataset

if TYPE_CHECKING:
    from pycocotools.coco import _Annotation

    from todd.tasks.object_detection import BBox, FlattenBBoxesXYWH

URL = 'http://images.cocodataset.org/'
Split = Literal['train', 'val']
Year = Literal[2014, 2017]


def coco_url(split: Split, year: Year, id_: int) -> str:
    return f'{URL}{split}{year}/{id_:012d}.jpg'


class BaseKeys(KeysProtocol[str], ABC):

    def __init__(self, image_ids: Iterable[int], suffix: str) -> None:
        self._image_ids = list(image_ids)
        self._suffix = suffix

    @property
    def image_ids(self) -> list[int]:
        return self._image_ids

    def __len__(self) -> int:
        return len(self._image_ids)

    @abstractmethod
    def _getitem(self, image_id: int) -> str:
        pass

    def __getitem__(self, index: int) -> str:
        item = self._getitem(self._image_ids[index])
        return item.removesuffix(f'.{self._suffix}')


class Keys(BaseKeys):

    def __init__(self, coco: COCO, *args, **kwargs) -> None:
        self._coco = coco
        super().__init__(coco.getImgIds(), *args, **kwargs)

    def _getitem(self, image_id: int) -> str:
        image, = self._coco.loadImgs(image_id)
        return image['file_name']


@dataclass(frozen=True)
class Annotation:
    mask: torch.Tensor
    area: float
    is_crowd: bool
    bbox: 'BBox'
    category: int

    @classmethod
    def load(
        cls,
        coco: COCO,
        annotation: '_Annotation',
        categories: Mapping[int, int],
    ) -> Self:
        mask = (
            torch.from_numpy(coco.annToMask(annotation))
            if 'segmentation' in annotation else torch.zeros(1)
        )
        return cls(
            mask,
            annotation['area'],
            bool(annotation['iscrowd']),
            cast('BBox', annotation['bbox']),
            categories[annotation['category_id']],
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
            Annotation.load(coco, annotation, categories)
            for annotation in annotations
        )

    @property
    def masks(self) -> torch.Tensor:
        return torch.stack([annotation.mask for annotation in self])

    @property
    def areas(self) -> torch.Tensor:
        return torch.tensor([annotation.area for annotation in self])

    @property
    def is_crowd(self) -> torch.Tensor:
        return torch.tensor([annotation.is_crowd for annotation in self])

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


APIType = TypeVar('APIType')  # pylint: disable=invalid-name
DataType = TypeVar('DataType')  # pylint: disable=invalid-name


class BaseDataset(
    PILDataset[DataType],
    Generic[APIType, DataType],
    ABC,
):
    SUFFIX = 'jpg'

    def __init__(
        self,
        *args,
        load_annotations: bool = True,
        api: APIType,
        **kwargs,
    ) -> None:
        self._load_annotations = load_annotations
        self._api = api
        super().__init__(*args, **kwargs)

    @property
    def api(self) -> APIType:
        return self._api


class T(TypedDict):
    id_: str
    image: torch.Tensor
    annotations: Annotations


@DatasetRegistry.register_()
class COCODataset(BaseDataset[COCO, T]):
    _keys: Keys

    DATA_ROOT = pathlib.Path('data/coco')
    ANNOTATIONS_ROOT = DATA_ROOT / 'annotations'

    def __init__(
        self,
        *args,
        split: Split,
        year: Year = 2017,
        access_layer: PILAccessLayer | None = None,
        annotations_file: pathlib.Path | str | None = None,
        **kwargs,
    ) -> None:
        split_year = f'{split}{year}'
        if access_layer is None:
            access_layer = PILAccessLayer(
                data_root=str(self.DATA_ROOT),
                task_name=split_year,
                suffix=self.SUFFIX,
            )
        if annotations_file is None:
            annotations_file = (
                self.ANNOTATIONS_ROOT / f'instances_{split_year}.json'
            )

        coco = COCO(annotations_file)

        self._categories = {
            category_id: i
            for i, category_id in enumerate(coco.getCatIds())
        }

        super().__init__(*args, api=coco, access_layer=access_layer, **kwargs)

    def build_keys(self) -> Keys:
        return Keys(self._api, self.SUFFIX)

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
