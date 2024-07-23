__all__ = [
    'LVISDataset',
]

from typing import Any, cast

from todd.datasets.access_layers.pil import PILAccessLayer

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


@DatasetRegistry.register_()
class LVISDataset(COCODataset):

    def __init__(
        self,
        *args,
        access_layer: PILAccessLayer | None = None,
        **kwargs,
    ) -> None:
        if access_layer is None:
            access_layer = PILAccessLayer(
                data_root=str(self.DATA_ROOT),
                suffix=self.SUFFIX,
            )
        super().__init__(*args, access_layer=access_layer, **kwargs)

    def build_keys(self) -> COCOKeys:
        return Keys(self._coco, self.SUFFIX)
