import typing

from todd.datasets import SATINDataset
from todd.datasets.satin import Split


def main() -> None:
    for split in typing.get_args(Split):
        SATINDataset(split=split)


if __name__ == '__main__':
    main()
