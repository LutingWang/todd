import os
from setuptools import find_packages, setup

import pkg_resources

import todd

pwd = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(pwd, "README.md")) as readme:
    long_description = readme.read()

with open(os.path.join(pwd, "requirements.txt")) as requirements:
    install_requires = [
        str(r) for r in pkg_resources.parse_requirements(requirements)
    ]

setup(
    name=todd.__name__,
    version=todd.__version__,
    description=todd.__doc__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=todd.__author__,
    author_email='wangluting@buaa.edu.cn',
    url='https://github.com/LutingWang/todd',
    packages=find_packages(),
    install_requires=install_requires,
)
