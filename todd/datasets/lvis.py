__all__ = [
    'LVISDataset',
]

import pathlib
from typing import TYPE_CHECKING, Any, Literal, cast

from pycocotools.coco import COCO

from ..registries import DatasetRegistry
from .access_layers import PILAccessLayer
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
        version: Literal['v0.5', 'v1'] | None = 'v1',
        access_layer: PILAccessLayer | None = None,
        annotations_file: pathlib.Path | str | None = None,
        **kwargs,
    ) -> None:
        if version is None:
            assert annotations_file is not None
        if access_layer is None:
            access_layer = PILAccessLayer(
                data_root=str(self.DATA_ROOT),
                suffix=self.SUFFIX,
            )
        if annotations_file is None:
            annotations_file = (
                self.ANNOTATIONS_ROOT / f'lvis_{version}_{split}.json'
            )
        super().__init__(
            *args,
            split=split,
            year=None,
            access_layer=access_layer,
            annotations_file=annotations_file,
            **kwargs,
        )
