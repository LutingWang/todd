import argparse
import os
import pathlib
from typing import TypeVar, cast

import einops
import numpy as np
import numpy.typing as npt
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from todd.datasets import BaseDataset
from todd.datasets.access_layers import PthAccessLayer
from todd.loggers import logger
from todd.tasks.natural_language_processing import BPETrainer
from todd.tasks.natural_language_processing.bpe import TokenSequence
from todd.utils import Store

# TODO

KT = str
VT = npt.NDArray[np.int64]

ModuleType = TypeVar('ModuleType', bound=nn.Module)  # noqa: E501 pylint: disable=invalid-name


class OursTokenDataset(BaseDataset[torch.Tensor, KT, VT]):

    def __init__(self, *args, data_root: str, **kwargs) -> None:
        access_layer: PthAccessLayer[np.int64] = \
            PthAccessLayer(data_root=data_root)
        super().__init__(*args, access_layer=access_layer, **kwargs)

    def __getitem__(self, index: int) -> torch.Tensor:
        _, item = self._access(index)
        tokens = item['tokens']
        tokens = einops.rearrange(tokens, 'b h w -> b (h w)')
        return tokens


def collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    name: str = args.name

    cpu = os.cpu_count() or 1
    dataset = OursTokenDataset(data_root='work_dir')
    dataloader: DataLoader[torch.Tensor] = DataLoader(
        dataset,
        cpu,
        num_workers=cpu,
        collate_fn=collate_fn,
    )
    logger.info("Loading...")
    token_sequences: list[TokenSequence] = sum(
        (
            cast(torch.Tensor, batch).tolist()
            for batch in tqdm.tqdm(dataloader)
        ),
        [],
    )

    bpe_trainer = BPETrainer(
        token_sequences=token_sequences,
        codebook_size=128,
        max_size=16384,
    )
    bpe_trainer.start()
    bpe, new_token_sequences = bpe_trainer.train()
    bpe_trainer.join()

    logger.info(
        "%d/%d",
        sum(map(len, new_token_sequences)),
        sum(map(len, token_sequences)),
    )

    work_dir = pathlib.Path('work_dirs')
    if Store.DRY_RUN:
        work_dir = work_dir / 'dry_run'
    work_dir = work_dir / name
    work_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bpe, work_dir / 'bpe.pth')
    torch.save(new_token_sequences, work_dir / 'token_sequences.pth')


if __name__ == '__main__':
    main()
