import os

import pkg_resources
from setuptools import find_packages, setup


with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as requirements:
    install_requires = [str(r) for r in pkg_resources.parse_requirements(requirements)]

setup(
    name='distilltools',
    packages=find_packages(),
    install_requires=install_requires,
)