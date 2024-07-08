import os
import pathlib
import tarfile
from typing import Any, TypedDict, cast

import numpy as np
import numpy.typing as npt
import scipy.io
import tqdm

from todd.patches.py import json_dump

data_root = pathlib.Path('data/imagenet')


class Synset(TypedDict):
    WNID: str
    words: str
    gloss: str
    num_children: int
    children: list[int]
    wordnet_height: int
    num_train_images: int


Synsets = dict[int, Synset]


class Annotation(TypedDict):
    name: str
    synset_id: int


def parse_synset(array: npt.NDArray[Any]) -> tuple[int, Synset]:
    field_names = array.dtype.names
    assert field_names is not None
    synset = dict(zip(field_names, array))

    v: npt.NDArray[Any]
    for k, v in synset.items():
        synset[k] = v.tolist() if k == 'children' else v.item()

    synset_id = synset.pop('ILSVRC2012_ID')
    return synset_id, cast(Synset, synset)


def parse_synsets() -> Synsets:
    meta = scipy.io.loadmat(
        str(data_root / 'ILSVRC2012_devkit_t12' / 'data' / 'meta.mat'),
    )
    return dict(parse_synset(synset[0]) for synset in meta['synsets'])


def train_synset(wnid: str, tar: tarfile.TarFile) -> None:
    prefix = wnid + '_'
    path = data_root / 'train' / wnid
    path.mkdir(parents=True, exist_ok=True)
    for image in tqdm.tqdm(tar.getmembers(), leave=False):
        assert image.isfile() and image.name.startswith(prefix)
        image_file = tar.extractfile(image)
        assert image_file is not None
        image_path = path / image.name.removeprefix(prefix)
        image_path.write_bytes(image_file.read())


def train(tar: tarfile.TarFile) -> None:
    for synset in tqdm.tqdm(tar.getmembers()):
        assert synset.isfile() and synset.name.endswith('.tar')
        wnid, _ = os.path.splitext(synset.name)
        synset_file = tar.extractfile(synset)
        with tarfile.open(fileobj=synset_file) as synset_tar:
            train_synset(wnid, synset_tar)


def val(tar: tarfile.TarFile, synsets: Synsets) -> None:
    gt_path = (
        data_root / 'ILSVRC2012_devkit_t12' / 'data'
        / 'ILSVRC2012_validation_ground_truth.txt'
    )
    gt = np.loadtxt(gt_path, np.uint16).tolist()

    for synset_id in set(gt):
        path = data_root / 'val' / synsets[synset_id]['WNID']
        path.mkdir(parents=True, exist_ok=True)

    for image, synset_id in zip(tqdm.tqdm(tar.getmembers()), gt):
        assert image.isfile() and image.name.endswith('.JPEG')
        path = data_root / 'val' / synsets[synset_id]['WNID']
        image_file = tar.extractfile(image)
        assert image_file is not None
        image_path = (
            path / image.name.removeprefix('ILSVRC2012_val_').lstrip('0')
        )
        image_path.write_bytes(image_file.read())


def annotations(split_name: str, synsets: Synsets) -> None:
    split_root = data_root / split_name
    json_dump(
        [
            Annotation(name=image.name, synset_id=synset_id)
            for synset_id, synset in synsets.items()
            if (synset_path := split_root / synset['WNID']).exists()
            for image in synset_path.iterdir()
        ],
        data_root / 'annotations' / f'{split_name}.json',
    )


def main() -> None:
    synsets = parse_synsets()
    json_dump(synsets, data_root / 'synsets.json')
    with tarfile.open(data_root / 'ILSVRC2012_img_train.tar') as tar:
        train(tar)
    with tarfile.open(data_root / 'ILSVRC2012_img_val.tar') as tar:
        val(tar, synsets)
    annotations('train', synsets)
    annotations('val', synsets)


if __name__ == '__main__':
    main()
