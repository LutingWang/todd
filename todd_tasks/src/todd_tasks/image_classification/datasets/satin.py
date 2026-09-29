__all__ = [
    'SATINDataset',
]

import io
from typing import Any, Iterator, Literal, TypedDict

import datasets
import torch
import torchvision.transforms.functional as F
from PIL import Image

from todd_torch.datasets import BaseAccessLayer, BaseDataset, IndexKeys
from todd_torch.patches.pil import convert_rgb
from todd_torch.registries import DatasetRegistry


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


class SATINAccessLayer(BaseAccessLayer[int, dict[str, Any]]):

    def __init__(self, *args, split: Split, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dataset = datasets.load_dataset(
            'jonathan-roberts1/satin',
            name=split,
            split=datasets.Split.TRAIN,
            trust_remote_code=True,
        )

    @property
    def exists(self) -> bool:
        return True

    def touch(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self)))

    def __getitem__(self, key: int) -> dict[str, Any]:
        return self._dataset[key]

    def __delitem__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def __setitem__(self, *args, **kwargs) -> None:
        raise NotImplementedError


@DatasetRegistry.register_()
class SATINDataset(BaseDataset[T, int, dict[str, Any]]):

    def __init__(
        self,
        *args,
        split: Split,
        access_layer: SATINAccessLayer | None = None,
        **kwargs,
    ) -> None:
        if access_layer is None:
            access_layer = SATINAccessLayer(split=split)

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
