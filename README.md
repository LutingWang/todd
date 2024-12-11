# Toolkit for Object Detection Distillation

```text
_/_/_/_/_/                _/        _/
   _/      _/_/      _/_/_/    _/_/_/
  _/    _/    _/  _/    _/  _/    _/
 _/    _/    _/  _/    _/  _/    _/
_/      _/_/      _/_/_/    _/_/_/
```

[![docs](https://readthedocs.org/projects/toddai/badge/?version=latest)](https://toddai.readthedocs.io/en/latest/?badge=latest)
[![lint](https://github.com/LutingWang/todd/actions/workflows/lint.yaml/badge.svg)](https://github.com/LutingWang/todd/actions/workflows/lint.yaml)
[![test](https://github.com/LutingWang/todd/actions/workflows/test.yaml/badge.svg)](https://github.com/LutingWang/todd/actions/workflows/test.yaml)
[![publish](https://github.com/LutingWang/todd/actions/workflows/publish.yaml/badge.svg)](https://github.com/LutingWang/todd/actions/workflows/publish.yaml)

[![codecov](https://codecov.io/gh/LutingWang/todd/branch/main/graph/badge.svg?token=BHDPCKVM1T)](https://codecov.io/gh/LutingWang/todd)
[![PyPI](https://img.shields.io/pypi/v/todd_ai)](https://pypi.org/project/todd-ai/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/todd_ai)](https://pypi.org/project/todd-ai/)
[![wakatime](https://wakatime.com/badge/github/LutingWang/todd.svg)](https://wakatime.com/badge/github/LutingWang/todd)

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](.github/CODE_OF_CONDUCT.md)

## Installation

Prerequisites:

- torch
- torchvision
- torchaudio

```bash
pip install git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021
pip install todd_ai
```

Optional dependencies:

```bash
mim install mmcv
```

## Developer Guides

```bash
pip install .\[optional,dev,lint,doc,test\]
```

TODO

1. Complete repr, especially for runners
2. add Any to all missing typing fields
3. @pytest.fixture(autouse=True)
4. finish docs/source/api_reference/tasks/bpe.py and docs/source/pretrained/stable_diffusion.py
5. remove todd.scripts
6. update docformatter to v1.8 in pre-commit-config and remove bandit B614 from pyproject.toml
