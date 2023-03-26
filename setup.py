import os
import pathlib

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

extensions = []
if os.getenv('EXTENSIONS'):
    root = pathlib.Path('todd/extensions')
    for path in root.iterdir():
        name = '.'.join(path.parts)
        sources = list(map(str, path.glob('*.cpp')))
        cpp_extension = CppExtension(name + '.cpp', sources)
        extensions.append(cpp_extension)

setup(
    ext_modules=extensions,
    cmdclass={
        'build_ext': BuildExtension,
    },
)
