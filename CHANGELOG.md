## v0.2.4a0 (2022-10-06)

### Feat

- workflow specs and checkpoint

## v0.2.3 (2022-10-05)

### Feat

- multiple updates

## v0.2.3a15 (2022-10-04)

### Feat

- **distillers**: refactor meta class build

## v0.2.3a14 (2022-09-21)

### Feat

- **utils**: lr scheduler

## v0.2.3a13 (2022-09-21)

### Feat

- **utils**: build optimizers

## v0.2.3a12 (2022-09-19)

### Feat

- **base**: bboxes support for empty

## v0.2.3a11 (2022-09-11)

### Feat

- **base**: change logging format
- **base**: bboxes support for inputs neither tensor nor ndarray

### Refactor

- **base**: move dict action from _extensions to configs

## v0.2.3a10 (2022-09-01)

### Fix

- **base**: subprocess run in Python3.8 does not have text and capture_output options

## v0.2.3a9 (2022-09-01)

### Feat

- **reproduction**: simplify model freeze api

## v0.2.3a8 (2022-08-31)

### Fix

- **base**: config list items should not be sorted

## v0.2.3a7 (2022-08-30)

### Feat

- **base**: add git APIs

### Fix

- **base**: DictAction collide with required keyword

## v0.2.3a6 (2022-08-29)

### Feat

- **base**: debug base class

## v0.2.3a5 (2022-08-27)

### Feat

- **base**: log file can be manually specified

### Fix

- **base**: log file does not relate to get_rank()

## v0.2.3a4 (2022-08-25)

### Feat

- **base**: checkpoints load open mmlab models signiture

## v0.2.3a3 (2022-08-25)

### Feat

- **base**: config merge with list indices

## v0.2.3a2 (2022-08-25)

### Fix

- **base**: config hasattr bug

## v0.2.3a1 (2022-08-25)

### Fix

- **base**: config merge bug

## v0.2.3a0 (2022-08-25)

### Feat

- **base**: add diff to config
- **base**: rewrite config to conform mmcv

## v0.2.2 (2022-08-23)

### Fix

- **base**: config cannot inherit UserDict

## v0.2.1 (2022-08-23)

### Feat

- **base**: config dump
- **base**: configs
- **adapts**: register all adaptations

## v0.2.0 (2022-08-14)

### Feat

- **schedulers**: chained scheduler with stand-along value
- **base**: debug
- **base**: debug options

## v0.1.6 (2022-07-31)

## v0.1.5 (2022-07-31)

### Refactor

- **reproduction**: model no grad or eval

### Fix

- **reproduction**: set seed with str may exceed 2**30

## v0.1.4 (2022-07-26)

### Fix

- **base**: patch

## v0.1.3 (2022-07-26)

### Fix

- **base**: patch attrs

### Feat

- remove mmcv dependencies

## v0.1.3a0 (2022-07-25)

## v0.1.2 (2022-07-25)

### Refactor

- distill

## v0.1.1 (2022-07-24)

## v0.1.0 (2022-07-24)

### Feat

- attr

### Fix

- workflow publish
- workflow publish
- workflow publish
- workflow publish

## v0.0.1 (2022-07-21)

### Fix

- compat python 3.6
- workflows
- github workflow keeps crash
- docs
- dict tensor adapt
- github workflow and rtd
- optionally patch builtins.breakpoint
- read the docs cannot import todd because of wrong conf.py
- bboxes expand image_shape (h,w) instead of (w,h)
- freeze_model with no args
- MSE2DLoss
- multilevel mask
- fgd loss
- true div for masks adapt
- rename weight to mask
- freeze offline teachers
- visualize
- mixin distiller
- hooks.__init__
- remove closure of hooks
- trackings register tensor
- import

### Feat

- setattr_temp
- convert logs
- workflow
- mypy workflows
- mypy workflows
- registry
- registry build with abc base
- registry build
- dict action
- colorize log
- losses with bound
- coveragerc
- flake8
- flake8
- registry
- ce loss
- scheduler as float
- get rand and world_size
- compat python 3.6
- distiller decorator
- README
- github workflow
- github workflow build
- github workflow build
- github workflow build
- github workflow build
- github workflow
- github workflow
- coverage
- pytest
- ansi color logger
- frs mask
- focal loss
- datasets
- datasets and criterions
- support denseclip
- compat functools cache
- sync code
- compat python3.6 for missing cached_properties
- gdet
- sgfi loss
- ckd loss
- defeat loss pool
- mask ceil mode
- defeat loss
- fgfi mask
- lfs
- fgd
- distiller.reset
- compare configs
- compare configs
- move schedualer to top level
- schedualers
- enable dict ids
- visualize
- rename decorator mixin
- convert adapt dict to adapt list
- dict tensor
- generalized tensors
- detach context
- merge hooks.builder and hooks.modules
- rename multidetach to list detach
