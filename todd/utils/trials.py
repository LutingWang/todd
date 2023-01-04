__all__ = [
    'init_trial',
    'TrialStore',
]

from datetime import datetime
from pathlib import Path

import torch

from ..base import Store, StoreMeta


def init_trial(
    root: str = 'work_dirs',
    trial_name: str = 'debug',
) -> Path:
    trial_root = Path(root).resolve(strict=True) / trial_name
    trial_root.mkdir(exist_ok=True)
    log_file = datetime.now().strftime("%Y%m%dT%H%M%S%f") + '.log'
    Store.LOG_FILE = str(trial_root / log_file)
    return trial_root


class TrialStore(metaclass=StoreMeta):
    CPU: bool
    CUDA: bool

    DRY_RUN: bool
    TRAIN_WITH_VAL_DATASET: bool


if not TrialStore.CPU and not TrialStore.CUDA:
    if torch.cuda.is_available():
        TrialStore.CUDA = True
    else:
        TrialStore.CPU = True
assert TrialStore.CPU + TrialStore.CUDA == 1

if TrialStore.CPU:
    TrialStore.DRY_RUN = True
    TrialStore.TRAIN_WITH_VAL_DATASET = True

    try:
        import mmcv.cnn
        mmcv.cnn.NORM_LAYERS.register_module(
            name='SyncBN',
            force=True,
            module=torch.nn.BatchNorm2d,
        )
    except Exception:
        pass
