__all__ = [
    'collect_env',
    'git_commit_id',
    'nvidia_smi',
]

import importlib.util
import os
import platform
import subprocess
import sys

import torch
from torch.utils.cpp_extension import CUDA_HOME


def git_commit_id(verbose: bool = False) -> str | None:
    args = 'git rev-parse HEAD' if verbose else 'git rev-parse --short HEAD'
    try:
        return subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return os.getenv('GIT_COMMIT_ID')


def nvidia_smi(verbose: bool = False) -> str | None:
    args = 'nvidia-smi -q' if verbose else 'nvidia-smi -L'
    try:
        return subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def collect_env(verbose: bool = False) -> dict[str, str | None]:
    env: dict[str, str | None] = dict()

    # system info
    env['Platform'] = platform.platform()
    env['Nvidia SMI'] = nvidia_smi(verbose)

    # versions
    env['Python version'] = sys.version
    env['PyTorch version'] = torch.__version__
    if importlib.util.find_spec('torchvision'):
        import torchvision
        env['TorchVision version'] = torchvision.__version__
    if importlib.util.find_spec('cv2'):
        import cv2
        env['OpenCV version'] = cv2.__version__
    from .. import __version__
    env['Todd version'] = __version__

    # environment variables
    env['CUDA_HOME'] = CUDA_HOME

    env['Git commit ID'] = git_commit_id(verbose)

    return env
