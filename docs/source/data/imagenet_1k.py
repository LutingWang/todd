import os
import tarfile
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy.io
import tqdm

from todd.patches import json_dump
from todd_tasks.image_classification.datasets import ImageNetDataset
from todd_tasks.image_classification.datasets.imagenet import (
    Annotation,
    Synset,
    Synsets,
)


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
        str(
            ImageNetDataset.DATA_ROOT / 'ILSVRC2012_devkit_t12' / 'data'
            / 'meta.mat',
        ),
    )
    return dict(parse_synset(synset[0]) for synset in meta['synsets'])


def train_synset(wnid: str, tar: tarfile.TarFile) -> None:
    prefix = wnid + '_'
    directory = ImageNetDataset.DATA_ROOT / 'train' / wnid
    directory.mkdir(parents=True, exist_ok=True)
    for image_file in tqdm.tqdm(tar.getmembers(), leave=False):
        assert image_file.isfile() and image_file.name.startswith(prefix)
        image_stream = tar.extractfile(image_file)
        assert image_stream is not None
        output_file = directory / image_file.name.removeprefix(prefix)
        output_file.write_bytes(image_stream.read())


def train(tar: tarfile.TarFile) -> None:
    for synset in tqdm.tqdm(tar.getmembers()):
        assert synset.isfile() and synset.name.endswith('.tar')
        wnid, _ = os.path.splitext(synset.name)
        synset_stream = tar.extractfile(synset)
        with tarfile.open(fileobj=synset_stream) as synset_tar:
            train_synset(wnid, synset_tar)


def val(tar: tarfile.TarFile, synsets: Synsets) -> None:
    annotation_file = (
        ImageNetDataset.DATA_ROOT / 'ILSVRC2012_devkit_t12' / 'data'
        / 'ILSVRC2012_validation_ground_truth.txt'
    )
    annotation = np.loadtxt(annotation_file, np.uint16).tolist()

    for synset_id in set(annotation):
        directory = ImageNetDataset.DATA_ROOT / 'val' / synsets[synset_id][
            'WNID']
        directory.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        tar.getmembers(),
        key=lambda image_file: image_file.name,
    )
    for image_file, synset_id in zip(tqdm.tqdm(image_files), annotation):
        assert (
            image_file.isfile()
            and image_file.name.endswith(f'.{ImageNetDataset.SUFFIX}')
        )
        directory = ImageNetDataset.DATA_ROOT / 'val' / synsets[synset_id][
            'WNID']
        image_stream = tar.extractfile(image_file)
        assert image_stream is not None
        output_file = (
            directory
            / image_file.name.removeprefix('ILSVRC2012_val_').lstrip('0')
        )
        output_file.write_bytes(image_stream.read())


def annotations(split: str, synsets: Synsets) -> None:
    split_dir = ImageNetDataset.DATA_ROOT / split
    synset_dirs = {
        synset_id: split_dir / synset['WNID']
        for synset_id, synset in synsets.items()
    }
    synset_dirs = {
        synset_id: synset_dir
        for synset_id, synset_dir in synset_dirs.items()
        if synset_dir.exists()
    }
    synset_dirs = dict(sorted(synset_dirs.items(), key=lambda x: x[1]))
    json_dump(
        [
            Annotation(name=image_file.name, synset_id=synset_id)
            for synset_id, synset_dir in synset_dirs.items()
            for image_file in synset_dir.iterdir()
        ],
        ImageNetDataset.ANNOTATIONS_DIR / f'{split}.json',
    )


def main() -> None:
    synsets = parse_synsets()
    json_dump(synsets, ImageNetDataset.SYNSETS_FILE)
    with tarfile.open(
        ImageNetDataset.DATA_ROOT / 'ILSVRC2012_img_train.tar',
    ) as tar:
        train(tar)
    with tarfile.open(
        ImageNetDataset.DATA_ROOT / 'ILSVRC2012_img_val.tar',
    ) as tar:
        val(tar, synsets)
    annotations('train', synsets)
    annotations('val', synsets)


if __name__ == '__main__':
    main()
