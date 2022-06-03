import os
from setuptools import find_packages, setup

import pkg_resources

with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as requirements:
    install_requires = [str(r) for r in pkg_resources.parse_requirements(requirements)]

setup(
    name='todd',
    packages=find_packages(),
    install_requires=install_requires,
)