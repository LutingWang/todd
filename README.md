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
[![PyPI - Downloads](https://img.shields.io/pypi/dm/todd_ai)]((https://pypi.org/project/todd-ai/))
[![wakatime](https://wakatime.com/badge/github/LutingWang/todd.svg)](https://wakatime.com/badge/github/LutingWang/todd)

## Installation

Prerequisites:

- torch
- torchvision
- mmcv/mmcv_full (optional)

```shell
pip install todd_ai
```

## Developer Guides

```bash
pip install .\[dev,doc,test\]
```

TODO

1. Complete repr, especially for runners
2. mypy supports recursive types
3. add a script to prepare imagenet dataset
4. add Any to all missing typing fields
5. make loss registry a sub-registry of model registry
6. refactor registries
7. remove odpsrun
