__all__ = [
    'COCODataset',
]

import pathlib
from collections import UserList
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, Literal, Mapping, TypedDict, cast
from typing_extensions import Self

import torch
from pycocotools.coco import COCO

from ..registries import DatasetRegistry
from .access_layers import PILAccessLayer
from .pil import PILDataset

if TYPE_CHECKING:
    from pycocotools.coco import _Annotation

    from todd.tasks.object_detection import BBox, BBoxesXYXY

Split = Literal['train', 'val']


class Keys:

    def __init__(self, coco: COCO, suffix: str) -> None:
        self._coco = coco
        self._suffix = suffix
        self._image_ids = coco.getImgIds()

    @property
    def image_ids(self) -> list[int]:
        return self._image_ids

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, index: int) -> str:
        image_id = self._image_ids[index]
        image, = self._coco.loadImgs(image_id)
        return image['file_name'].removesuffix(f'.{self._suffix}')

    def __iter__(self) -> Generator[str, None, None]:
        for i in range(len(self)):
            yield self[i]


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
        return cls(
            torch.from_numpy(coco.annToMask(annotation)),
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
    def bboxes(self) -> 'BBoxesXYXY':
        from todd.tasks.object_detection import BBoxesXYXY
        return BBoxesXYXY(
            torch.tensor([annotation.bbox for annotation in self]),
        )

    @property
    def categories(self) -> torch.Tensor:
        return torch.tensor([annotation.category for annotation in self])


class T(TypedDict):
    id_: str
    image: torch.Tensor
    annotations: Annotations


@DatasetRegistry.register_()
class COCODataset(PILDataset[T]):
    _keys: Keys

    DATA_ROOT = pathlib.Path('data/coco')
    ANNOTATIONS_ROOT = DATA_ROOT / 'annotations'
    YEAR = 2017
    SUFFIX = 'jpg'

    def __init__(
        self,
        *args,
        split: Split,
        access_layer: PILAccessLayer | None = None,
        annotations_file: pathlib.Path | str | None = None,
        load_annotations: bool = True,
        **kwargs,
    ) -> None:
        self._load_annotations = load_annotations

        split_year = f'{split}{self.YEAR}'

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
        elif isinstance(annotations_file, str):
            annotations_file = pathlib.Path(annotations_file)

        coco = COCO(annotations_file)
        self._coco = coco

        self._categories = {
            category_id: i
            for i, category_id in enumerate(coco.getCatIds())
        }

        super().__init__(*args, access_layer=access_layer, **kwargs)

    def build_keys(self) -> Keys:
        return Keys(self._coco, self.SUFFIX)

    @property
    def coco(self) -> COCO:
        return self._coco

    def __getitem__(self, index: int) -> T:
        key, image = self._access(index)

        # NOTE: annotations are not transformed
        tensor = self._transform(image)

        image_id = self._keys.image_ids[index]
        annotations = (
            Annotations.load(self._coco, image_id, self._categories)
            if self._load_annotations else Annotations()
        )
        return T(id_=key, image=tensor, annotations=annotations)
