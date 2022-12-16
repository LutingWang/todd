__all__ = [
    'init_trial',
    'TrialStore',
]

from datetime import datetime
from pathlib import Path

import torch.cuda

from ..base import Store, StoreMeta


def init_trial(
    root: str = 'work_dirs',
    trial_name: str = 'debug',
) -> Path:
    trial_root = Path(root).resolve(strict=True)
    trial_root /= trial_name
    trial_root.mkdir(exist_ok=True)
    log_file = datetime.now().strftime("%Y%m%dT%H%M%S%f") + '.log'
    Store.LOG_FILE = str(trial_root / log_file)
    return trial_root


class TrialStore(metaclass=StoreMeta):
    CPU: bool
    CUDA: bool

    DRY_RUN: bool
    TRAIN_WITH_VAL_DATASET: bool

    def __init__(self) -> None:
        if self.CPU or self.CUDA:
            assert self.CPU + self.CUDA == 1
        elif torch.cuda.is_available():
            self.CUDA = True
        else:
            self.CPU = True

        if self.CPU:
            self.DRY_RUN = True
            self.TRAIN_WITH_VAL_DATASET = True
