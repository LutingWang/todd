__all__ = [
    'LVISDataset',
]

import pathlib
from typing import TYPE_CHECKING, Any, cast

from pycocotools.coco import COCO

from ..registries import DatasetRegistry
from .coco import COCODataset, Split

if TYPE_CHECKING:
    from pycocotools.coco import _Image


class LVIS(COCO):

    def loadImgs(self, *args, **kwargs) -> list['_Image']:  # noqa: N802
        images = super().loadImgs(*args, **kwargs)
        for image in images:
            url: str = cast(dict[str, Any], image)['coco_url']
            image['file_name'] = url.removeprefix(
                'http://images.cocodataset.org/',
            )
        return images


@DatasetRegistry.register_()
class LVISDataset(COCODataset):
    DATA_ROOT = pathlib.Path('data/lvis')
    ANNOTATIONS_ROOT = DATA_ROOT / 'annotations'

    COCO_TYPE = LVIS

    def __init__(
        self,
        *args,
        split: Split,
        annotations_file: pathlib.Path | str | None = None,
        **kwargs,
    ) -> None:
        if annotations_file is None:
            annotations_file = (
                self.ANNOTATIONS_ROOT / f'lvis_v1_{split}.json'
            )
        super().__init__(
            *args,
            split=split,
            annotations_file=annotations_file,
            **kwargs,
        )
