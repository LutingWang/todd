__all__ = [
    'pytorch_version',
    'torchvision_version',
    'opencv_version',
    'cuda_home',
]

from todd.utils import EnvRegistry


@EnvRegistry.register_()
def pytorch_version(verbose: bool = False) -> str:
    import torch
    return torch.__version__


@EnvRegistry.register_()
def torchvision_version(verbose: bool = False) -> str:
    import torchvision
    return torchvision.__version__


@EnvRegistry.register_()
def opencv_version(verbose: bool = False) -> str:
    import cv2
    return cv2.__version__


@EnvRegistry.register_()
def cuda_home(verbose: bool = False) -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME
    return CUDA_HOME
