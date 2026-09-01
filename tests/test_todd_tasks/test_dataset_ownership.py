import todd_torch.datasets as torch_datasets
from todd_tasks.image_classification.datasets.imagenet import ImageNetDataset
from todd_tasks.image_classification.datasets.satin import SATINDataset
from todd_tasks.image_generation.datasets.laion_aesthetics import (
    LAIONAestheticsDataset,
)
from todd_tasks.image_segmentation.datasets.sa_med2d import SAMed2DDataset


def test_concrete_datasets_belong_to_tasks() -> None:
    assert ImageNetDataset.__module__.startswith(
        'todd_tasks.image_classification.',
    )
    assert SATINDataset.__module__.startswith(
        'todd_tasks.image_classification.',
    )
    assert LAIONAestheticsDataset.__module__.startswith(
        'todd_tasks.image_generation.',
    )
    assert SAMed2DDataset.__module__.startswith(
        'todd_tasks.image_segmentation.',
    )

    for name in (
        'ImageNetDataset',
        'SATINDataset',
        'LAIONAestheticsDataset',
        'SAMed2DDataset',
    ):
        assert not hasattr(torch_datasets, name)
