__all__ = [
    'EnvRegistry',
    'platform',
    'nvidia_smi',
    'python_version',
    'pytorch_version',
    'torchvision_version',
    'opencv_version',
    'todd_version',
    'cuda_home',
    'git_commit_id',
]

import importlib.util
import os
import subprocess

from .registries import Registry


class EnvRegistry(Registry):
    pass


@EnvRegistry.register('Platform')
def platform(verbose: bool = False) -> str | None:
    from platform import platform as _platform
    return _platform()


@EnvRegistry.register('NVIDIA SMI')
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


@EnvRegistry.register('Python version')
def python_version(verbose: bool = False) -> str | None:
    import sys
    return sys.version


@EnvRegistry.register('PyTorch version')
def pytorch_version(verbose: bool = False) -> str | None:
    import torch
    return torch.__version__


@EnvRegistry.register('TorchVision version')
def torchvision_version(verbose: bool = False) -> str | None:
    if not importlib.util.find_spec('torchvision'):
        return None
    import torchvision
    return torchvision.__version__


@EnvRegistry.register('OpenCV version')
def opencv_version(verbose: bool = False) -> str | None:
    if not importlib.util.find_spec('cv2'):
        return None
    import cv2
    return cv2.__version__


@EnvRegistry.register('Todd version')
def todd_version(verbose: bool = False) -> str | None:
    from .. import __version__
    return __version__


@EnvRegistry.register('CUDA_HOME')
def cuda_home(verbose: bool = False) -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME


@EnvRegistry.register('Git commit ID')
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
