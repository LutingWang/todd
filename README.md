# TODD

[![docs](https://readthedocs.org/projects/toddai/badge/?version=latest)](https://toddai.readthedocs.io/en/latest/?badge=latest)
[![lint](https://github.com/LutingWang/todd/actions/workflows/lint.yaml/badge.svg)](https://github.com/LutingWang/todd/actions/workflows/lint.yaml)
[![test](https://github.com/LutingWang/todd/actions/workflows/test.yaml/badge.svg)](https://github.com/LutingWang/todd/actions/workflows/test.yaml)
[![publish](https://github.com/LutingWang/todd/actions/workflows/publish.yaml/badge.svg)](https://github.com/LutingWang/todd/actions/workflows/publish.yaml)
[![codecov](https://codecov.io/gh/LutingWang/todd/branch/master/graph/badge.svg?token=BHDPCKVM1T)](https://codecov.io/gh/LutingWang/todd)
[![PyPI](https://img.shields.io/pypi/v/todd_ai)](https://pypi.org/project/todd-ai/)
[![wakatime](https://wakatime.com/badge/github/LutingWang/todd.svg)](https://wakatime.com/badge/github/LutingWang/todd)

Toolkit for Object Detection Distillation.

## Installation

Prerequisites:
- torch
- torchvision
- mmcv

```shell
pip install todd_ai
```

# Developer

## Commit

```shell
pre-commit install
pre-commit install -t commit-msg
```

Recommended to use [commitizen](https://github.com/commitizen-tools/commitizen) for commit message formatting.

```shell
cz c
```

## Docs

```shell
(cd docs && exec make html)
```

## Publish

```shell
git tag ${TAG}
git push --atomic origin master ${TAG}
```
