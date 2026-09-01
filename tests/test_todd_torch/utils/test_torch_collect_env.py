from todd.utils import EnvRegistry, collect_env_
from todd_torch.utils.collect_env import (
    cuda_home,
    opencv_version,
    pytorch_version,
    torchvision_version,
)


def test_collect_env() -> None:
    assert EnvRegistry[pytorch_version.__name__] is pytorch_version
    assert EnvRegistry[torchvision_version.__name__] is torchvision_version
    assert EnvRegistry[opencv_version.__name__] is opencv_version
    assert EnvRegistry[cuda_home.__name__] is cuda_home

    env = collect_env_()
    assert 'pytorch_version:' in env
    assert 'torchvision_version:' in env
    assert 'opencv_version:' in env
    assert 'cuda_home:' in env
