# pylint: disable=duplicate-code

__all__ = [
    'LVISDataset',
]

import pathlib
from collections import UserList
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypedDict, cast
from typing_extensions import Self

import torch
from lvis import LVIS
from lvis.boundary_utils import ann_to_mask

from ..registries import DatasetRegistry
from .access_layers import PILAccessLayer
from .coco import URL
from .coco import BaseDataset as BaseCOCODataset
from .coco import BaseKeys as COCOKeys

if TYPE_CHECKING:
    from todd.tasks.object_detection import BBox, FlattenBBoxesXYWH

Split = Literal['train', 'val', 'minival']
Version = Literal['v0.5', 'v1']


class Keys(COCOKeys):

    def __init__(self, lvis: LVIS, *args, **kwargs) -> None:
        self._lvis = lvis
        super().__init__(lvis.get_img_ids(), *args, **kwargs)

    def _getitem(self, image_id: int) -> str:
        image, = self._lvis.load_imgs([image_id])
        url: str = image['coco_url']
        return url.removeprefix(URL)


@dataclass(frozen=True)
class Annotation:
    area: float
    mask: torch.Tensor
    bbox: 'BBox'
    category: int

    @classmethod
    def load(
        cls,
        lvis: LVIS,
        annotation: Mapping[str, Any],
        categories: Mapping[int, int],
    ) -> Self:
        return cls(
            annotation['area'],
            torch.from_numpy(ann_to_mask(annotation, lvis.imgs)),
            cast('BBox', annotation['bbox']),
            categories[annotation['category_id']],
        )


class Annotations(UserList[Annotation]):

    @classmethod
    def load(
        cls,
        lvis: LVIS,
        image_id: int,
        categories: Mapping[int, int],
    ) -> Self:
        annotation_ids = lvis.get_ann_ids([image_id])
        annotations = lvis.load_anns(annotation_ids)
        return cls(
            Annotation.load(lvis, annotation, categories)
            for annotation in annotations
        )

    @property
    def areas(self) -> torch.Tensor:
        return torch.tensor([annotation.area for annotation in self])

    @property
    def masks(self) -> torch.Tensor:
        return torch.stack([annotation.mask for annotation in self])

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


class T(TypedDict):
    id_: str
    image: torch.Tensor
    annotations: Annotations


@DatasetRegistry.register_()
class LVISDataset(BaseCOCODataset[LVIS, T]):
    _keys: Keys

    DATA_ROOT = pathlib.Path('data/lvis')
    ANNOTATIONS_ROOT = DATA_ROOT / 'annotations'

    def __init__(
        self,
        *args,
        split: Split,
        version: Version = 'v1',
        access_layer: PILAccessLayer | None = None,
        annotations_file: pathlib.Path | str | None = None,
        **kwargs,
    ) -> None:
        if access_layer is None:
            access_layer = PILAccessLayer(
                data_root=str(self.DATA_ROOT),
                suffix=self.SUFFIX,
            )
        if annotations_file is None:
            annotations_file = (
                self.ANNOTATIONS_ROOT / f'lvis_{version}_{split}.json'
            )

        lvis = LVIS(annotations_file)

        self._categories = {
            category_id: i
            for i, category_id in enumerate(lvis.get_cat_ids())
        }

        super().__init__(*args, api=lvis, access_layer=access_layer, **kwargs)

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
