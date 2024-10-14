__all__ = [
    'SATINDataset',
]

import io
import pathlib
from typing import Any, Literal, TypedDict

import datasets
import torch
import torchvision.transforms.functional as F
from PIL import Image

from ..bases.configs import Config
from ..patches.pil import convert_rgb
from ..registries import DatasetRegistry
from .access_layers import HFAccessLayer
from .base import BaseDataset
from .index import IndexKeys
from .registries import AccessLayerRegistry


class T(TypedDict):
    id_: int
    image: torch.Tensor
    data: dict[str, Any]


Split = Literal['SAT-4', 'SAT-6', 'NASC-TG2', 'WHU-RS19', 'RSSCN7', 'RS_C11',
                'SIRI-WHU', 'EuroSAT', 'NWPU-RESISC45', 'PatternNet',
                'RSD46-WHU', 'GID', 'CLRS', 'Optimal-31',
                'Airbus-Wind-Turbines-Patches', 'USTC_SmokeRS',
                'Canadian_Cropland', 'Ships-In-Satellite-Imagery',
                'Satellite-Images-of-Hurricane-Damage',
                'Brazilian_Coffee_Scenes', 'Brazilian_Cerrado-Savanna_Scenes',
                'Million-AID', 'UC_Merced_LandUse_MultiLabel', 'MLRSNet',
                'MultiScene', 'RSI-CB256', 'AID_MultiLabel']


@DatasetRegistry.register_()
class SATINDataset(BaseDataset[T, int, dict[str, Any]]):
    DATA_ROOT = pathlib.Path('data/satin')

    def __init__(
        self,
        *args,
        split: Split,
        access_layer: HFAccessLayer | None = None,
        **kwargs,
    ) -> None:
        if access_layer is None:
            access_layer = AccessLayerRegistry.build(
                Config(
                    type=HFAccessLayer.__name__,
                    data_root=str(self.DATA_ROOT),
                    task_name=str(datasets.Split.TRAIN),
                    datasets=dict(
                        path='jonathan-roberts1/satin',
                        name=split,
                        trust_remote_code=True,
                    ),
                ),
            )

        super().__init__(*args, access_layer=access_layer, **kwargs)
        self._split = split

    def build_keys(self) -> IndexKeys:
        return IndexKeys(len(self._access_layer))

    def _transform(self, image: Image.Image) -> torch.Tensor:
        if self._transforms is None:
            return F.pil_to_tensor(image)
        return self._transforms(image)

    def __getitem__(self, index: int) -> T:
        key, data = self._access(index)
        image = data.pop('image')
        if not isinstance(image, Image.Image):
            image = Image.open(io.BytesIO(image['bytes']))
        image = convert_rgb(image)
        tensor = self._transform(image)
        return T(id_=key, image=tensor, data=data)
