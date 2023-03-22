import os
import pathlib

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

with_cpp_extensions = os.getenv('TODD_WITH_CPP_EXTENSIONS')
# with_cuda_extensions = (
#     torch.cuda.is_available() and os.getenv('TODD_WITH_CUDA_EXTENSIONS')
# )

root = pathlib.Path('todd/extensions')
cpp_extensions = [
    CppExtension(
        '.'.join(path.parts) + '.cpp',
        list(map(str, path.glob('*.cpp'))),
        extra_compile_args=['-g'],
    ) for path in root.iterdir()
] if with_cpp_extensions else []

setup(
    ext_modules=cpp_extensions,
    cmdclass={
        'build_ext': BuildExtension,
    },
)
