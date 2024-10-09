from setuptools import setup

setup(
    scripts=[
        'bin/auto_torchrun',
        'bin/pipenv_install',
        'todd/scripts/crop_border.py',
    ],
)
