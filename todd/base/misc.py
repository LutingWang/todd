__all__ = [
    'strict_zip_len',
    'which_git',
    'git_status',
    'git_commit_id',
]

import os
import shutil
import subprocess
from typing import Iterable, List, Sized

from ._extensions import get_logger


def strict_zip_len(iterable: Iterable[Sized]) -> int:
    """Length check for zip.

    Before Python 3.10, zip() stops when the shortest element is exhausted.
    This function checks the length of all elements and raises an error if
    they are not the same.

    Args:
        iterable: Iterable of sized elements.

    Returns:
        Length of the sized elements.

    Raises:
        ValueError: If the length of the sized elements are not the same.
    """
    lens = {len(e) for e in iterable}
    if len(lens) > 1:
        raise ValueError(f'Lengths of iterables are not the same: {lens}')
    return lens.pop()


def which_git() -> str:
    git = shutil.which('git')
    assert git is not None
    return git


def git_status() -> List[str]:
    process = subprocess.run(
        [which_git(), 'status', '--porcelain'],
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        get_logger().warning(process.stderr.strip())
    return process.stdout.split('\n')[:-1]


def git_commit_id() -> str:
    process = subprocess.run(
        [which_git(), 'rev-parse', '--short', 'HEAD'],
        capture_output=True,
        text=True,
    )
    if process.returncode == 0:
        return process.stdout.strip()
    get_logger().warning(process.stderr.strip())
    return os.getenv('GIT_COMMIT_ID', '')
