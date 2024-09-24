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
    'git_status',
    'collect_env_',
]

import argparse
import importlib.util
import os
import subprocess  # nosec B404

from ..bases.registries import Registry
from ..loggers import logger
from ..patches.py_ import run


class EnvRegistry(Registry):
    pass


@EnvRegistry.register_()
def platform(verbose: bool = False) -> str | None:
    from platform import platform as _platform
    return _platform(terse=not verbose)


@EnvRegistry.register_()
def nvidia_smi(verbose: bool = False) -> str | None:
    args = 'nvidia-smi -q' if verbose else 'nvidia-smi -L'
    try:
        return run(args)
    except subprocess.CalledProcessError:
        return None


@EnvRegistry.register_()
def python_version(verbose: bool = False) -> str | None:
    import sys
    return sys.version


@EnvRegistry.register_()
def pytorch_version(verbose: bool = False) -> str | None:
    import torch
    return torch.__version__


@EnvRegistry.register_()
def torchvision_version(verbose: bool = False) -> str | None:
    if not importlib.util.find_spec('torchvision'):
        return None
    import torchvision
    return torchvision.__version__


@EnvRegistry.register_()
def opencv_version(verbose: bool = False) -> str | None:
    if not importlib.util.find_spec('cv2'):
        return None
    import cv2
    return cv2.__version__


@EnvRegistry.register_()
def todd_version(verbose: bool = False) -> str | None:
    from .. import __version__
    return __version__


@EnvRegistry.register_()
def cuda_home(verbose: bool = False) -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME


@EnvRegistry.register_()
def git_commit_id(verbose: bool = False) -> str | None:
    args = 'git rev-parse HEAD' if verbose else 'git rev-parse --short HEAD'
    try:
        return run(args)
    except subprocess.CalledProcessError:
        return os.getenv('GIT_COMMIT_ID')


@EnvRegistry.register_()
def git_status(verbose: bool = False) -> str | None:
    args = 'git status'
    if not verbose:
        args += ' --porcelain'
    try:
        return run(args)
    except subprocess.CalledProcessError:
        return None


def collect_env_(*args, **kwargs) -> str:
    envs = ['']
    for k, v in EnvRegistry.items():
        env = v(*args, **kwargs)
        env = str(env).strip()
        if '\n' in env:
            env = '\n' + env
        envs.append(f'{k}: {env}')
    return '\n'.join(envs)


def collect_env_cli() -> None:
    parser = argparse.ArgumentParser(description="Collect Environment")
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()
    env = collect_env_(verbose=args.verbose)
    logger.info(env)
