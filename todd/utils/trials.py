__all__ = [
    'init_trial',
]

from datetime import datetime
from pathlib import Path

from ..base import globals_


def init_trial(
    root: str = 'work_dirs',
    trial_name: str = 'debug',
) -> Path:
    trial_root = Path(root).resolve(strict=True)
    trial_root /= trial_name
    trial_root.mkdir(exist_ok=True)
    log_file = datetime.now().strftime("%Y%m%dT%H%M%S%f") + '.log'
    globals_._log_file = trial_root / log_file
    return trial_root
