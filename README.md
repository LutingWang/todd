# TODD

[![docs](https://readthedocs.org/projects/toddai/badge/?version=latest)](https://toddai.readthedocs.io/en/latest/?badge=latest)
[![lint](https://github.com/LutingWang/todd/actions/workflows/lint.yaml/badge.svg)](https://github.com/LutingWang/todd/actions/workflows/lint.yaml)
[![test](https://github.com/LutingWang/todd/actions/workflows/test.yaml/badge.svg)](https://github.com/LutingWang/todd/actions/workflows/test.yaml)
[![publish](https://github.com/LutingWang/todd/actions/workflows/publish.yaml/badge.svg)](https://github.com/LutingWang/todd/actions/workflows/publish.yaml)
[![codecov](https://codecov.io/gh/LutingWang/todd/branch/master/graph/badge.svg?token=BHDPCKVM1T)](https://codecov.io/gh/LutingWang/todd)
[![PyPI](https://img.shields.io/pypi/v/todd_ai)](https://pypi.org/project/todd-ai/)
[![wakatime](https://wakatime.com/badge/github/LutingWang/todd.svg)](https://wakatime.com/badge/github/LutingWang/todd)

Toolkit for Object Detection Distillation

## Installation

Prerequisites:
- torch
- torchvision
- mmcv/mmcv_full (optional)

```shell
pip install todd_ai
```

# Developer Guides

## Installation

```bash
pip install .\[doc,dev,test,pre-commit\]
```

```bash
pre-commit install
pre-commit install -t commit-msg
```

Recommended to use [commitizen](https://github.com/commitizen-tools/commitizen)

```bash
brew install commitizen
```

## Docs

```shell
(cd docs && exec make html)
```

## Publish

To automatically bump the version based on the commits

```shell
cz bump -ch [--increment {major,minor,patch}] [-pr {alpha,beta,rc}]
```

If for any reason, the created tag and changelog were to be undone, this is the snippet:

```shell
git tag --delete ${TAG}
git reset HEAD~
git reset --hard HEAD
```

```shell
pytest && git push --atomic origin master ${TAG}
```

## TODO

1. Clean up registries
2. Complete repr
