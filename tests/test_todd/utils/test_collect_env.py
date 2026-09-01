import subprocess  # nosec B404
import sys


def test_collect_env_without_torch() -> None:
    code = '''
import sys


class TorchBlocker:

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname in ('cv2', 'torch', 'torchvision'):
            raise AssertionError(f'unexpected import: {fullname}')
        return None


sys.meta_path.insert(0, TorchBlocker)

from todd.utils.collect_env import EnvRegistry, collect_env_

env = collect_env_()
assert 'pytorch_version' not in EnvRegistry
assert 'torchvision_version' not in EnvRegistry
assert 'opencv_version' not in EnvRegistry
assert 'cuda_home' not in EnvRegistry
'''
    subprocess.run([sys.executable, '-c', code], check=True)  # nosec B603
