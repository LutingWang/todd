import typing

from todd_tasks.image_classification.datasets import SATINDataset
from todd_tasks.image_classification.datasets.satin import Split


def main() -> None:
    for split in typing.get_args(Split):
        SATINDataset(split=split)


if __name__ == '__main__':
    main()
