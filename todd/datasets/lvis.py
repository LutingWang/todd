__all__ = [
    'LVISMixin',
    'LVISDataset',
]

from typing import Any, cast

from ..registries import DatasetRegistry
from .coco import COCODataset
from .coco import Keys as COCOKeys


class Keys(COCOKeys):

    def __getitem__(self, index: int) -> str:
        image_id = self._image_ids[index]
        image, = self._coco.loadImgs(image_id)
        url: str = cast(dict[str, Any], image)['coco_url']
        filename = url.removeprefix('http://images.cocodataset.org/')
        return filename.removesuffix(f'.{self._suffix}')


class LVISMixin(COCODataset):

    def build_keys(self) -> COCOKeys:
        return COCOKeys(self._coco, self.SUFFIX)


@DatasetRegistry.register_()
class LVISDataset(LVISMixin, COCODataset):
    pass
