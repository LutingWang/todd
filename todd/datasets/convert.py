import argparse
from typing import List

from mmcv import Config, DictAction

from .base import BaseDataset
from .builder import DATASETS, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Convert datasets')
    parser.add_argument('--source', nargs='+', action=DictAction)
    parser.add_argument('--target', nargs='+', action=DictAction)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    source = build_dataset(args.source)
    target: BaseDataset = DATASETS.get(args.target.pop('type'))
    target.load_from(source, **args.target)


if __name__ == '__main__':
    main()
