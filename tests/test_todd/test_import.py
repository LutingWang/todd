import subprocess  # nosec B404
import sys


def test_import_without_optional_dependencies() -> None:
    code = '''
import sys


class OptionalDependencyBlocker:

    BLOCKED = (
        'cv2',
        'datasets',
        'lmdb',
        'lvis',
        'numpy',
        'PIL',
        'torch',
        'torchaudio',
        'torchvision',
    )

    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if any(
            fullname == name or fullname.startswith(f'{name}.')
            for name in cls.BLOCKED
        ):
            raise AssertionError(f'unexpected import: {fullname}')
        return None


sys.meta_path.insert(0, OptionalDependencyBlocker)

import todd
from todd.colors import BGR, HTML4, PALETTE, RGB, RGBA, YIQ, Color

assert 'lvis' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code], check=True)  # nosec B603
