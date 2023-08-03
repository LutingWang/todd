__all__ = [
    'platform',
    'nvidia_smi',
    'python_version',
    'pytorch_version',
    'torchvision_version',
    'opencv_version',
    'todd_version',
    'cuda_home',
    'git_commit_id',
    'git_status',
]

import importlib.util
import os
import subprocess


def platform(verbose: bool = False) -> str | None:
    from platform import platform as _platform
    return _platform(terse=not verbose)


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
        ).stdout
    except subprocess.CalledProcessError:
        return None


def python_version(verbose: bool = False) -> str | None:
    import sys
    return sys.version


def pytorch_version(verbose: bool = False) -> str | None:
    import torch
    return torch.__version__


def torchvision_version(verbose: bool = False) -> str | None:
    if not importlib.util.find_spec('torchvision'):
        return None
    import torchvision
    return torchvision.__version__


def opencv_version(verbose: bool = False) -> str | None:
    if not importlib.util.find_spec('cv2'):
        return None
    import cv2
    return cv2.__version__


def todd_version(verbose: bool = False) -> str | None:
    from .. import __version__
    return __version__


def cuda_home(verbose: bool = False) -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME


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
        ).stdout
    except subprocess.CalledProcessError:
        return os.getenv('GIT_COMMIT_ID')


def git_status(verbose: bool = False) -> str | None:
    args = 'git status'
    if not verbose:
        args += ' --porcelain'
    try:
        return subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
