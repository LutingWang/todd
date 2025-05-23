## v0.6.0 (2025-05-21)

### Feat

- statistician
- allow attention mask in eva
- eva_clip
- vit in_channels
- clip normalize arg
- llama
- new state dict converter
- finish some todos
- f5 tts
- use torchaudio instead of soundfile
- whisper
- v3det dataset
- patch torch cosine_similarity
- auto torchrun
- [WIP]
- ram
- prefetch dataloader
- ram
- **nlp**: revise clip tokenizer
- **nlp**: clip tokenizer
- **models**: clip
- **models**: vit uses patched sequential
- **models**: dinov2
- **models**: dino
- register transforms v2
- filename codec
- **strategies**: compile model
- object365
- **datasets**: unify coco and lvis
- **datasets**: coco do not transform annotations
- multiple
- **runners**: metrics
- **runners**: fix checkpoint
- multiple fixes
- satin dataset
- misc retry
- satin and sa_med2d datasets
- **datasets**: laion aesthetics
- **datasets**: add transforms property
- **tasks**: kd model adapt
- **tasks**: lmm image data
- **tasks**: lmm x2i data
- get config and new sequential
- **patches**: torch sequential
- **nlp**: bpe trainer
- **nlp**: bpe
- **od**: bboxes to mask
- configs and coco dataset
- **datasets**: update pil dataset
- **datasets**: coco and pil datasets
- **tasks**: large multimodal model
- **registries**: dataloder batch size in total
- **datasets**: transforms
- **tasks**: imagenet dataset and fid
- simplify distiller config
- **tasks**: points inside mask
- **registries**: change build_spec to build pre hooks
- **runners**: merge callbacks bind and init
- introduce priors to builders
- **runners**: move builders to build_spec
- **registries**: builders
- **tasks**: tap vid davis
- **tasks**: point tracking
- **utils**: serialize mixin
- worker init fn
- **tasks**: optical flow spring dataset
- optical flow dataset
- simplify store logic

### Fix

- missing import
- base convnext
- f5 tts tokenization
- **runners**: checkpoint callback creates broken latest symbolic link
- object365 to objects365
- **nlp**: clip tokenizer
- **datasets**: rename object365 to objects365
- **datasets**: lvis and coco annotations can be empty
- **datasests**: coco and lvis annotations are in XYWH format
- multiple
- multiple
- **datasets**: lvis
- **docs**: imagenet preprocessing
- satin dataset
- **registries**: fix yapf error with non-standard pyconfig
- **tasks**: x2i data repr
- **tasks**: bpe multiprocessing
- **registries**: yapf maybe unable to process config when build failed
- **tasks**: image generation fid
- **tasks**: kd pipeline
- **runners**: build hooks
- type annotations
- **tasks**: optical flow type
- numpy 2.0 is not supported by pytorch
- **tasks**: minor
- multiple

### Refactor

- state dict converter
- **conceptnet**: update
- **kd**: simplify pipeline build
- **registries**: process init args of composed callback in registry
- **configs**: move configs to base
- **utils**: nested_collection_utils
- **tasks**: distillation processors

## v0.5.1 (2024-06-01)

## v0.5.0 (2024-05-31)

### Feat

- png optical flow
- init

## v0.3.0 (2023-01-25)

### Feat

- bboxes
- **base**: merge into types
- **base**: refine logger

## v0.2.4 (2023-01-11)

### Feat

- runners, remove ITER and LOG_FILE
- **base**: configs
- new registry, workflow, store, and distiller API

### Fix

- **visuals**: numpy typing checks

## v0.2.4a5 (2022-11-17)

### Feat

- **base**: DictAction take :: (double colon) as sign for string

## v0.2.4a4 (2022-11-16)

### Feat

- **visuals**: refact activation
- **base**: more debug options
- odpsrun script

### Refactor

- chore files

## v0.2.4a3 (2022-10-19)

### Feat

- **base**: bbox scale

## v0.2.4a2 (2022-10-12)

### Feat

- **base**: debug mode can be forced in command line
- **base**: bbox expand image_shape renamed to image_wh to avoid ambiguity
- **base**: bboxes get indices
- **base**: bboxes filter should be done by user
- **base**: bboxes filter
- **base**: bbox now support convert
- **visuals**: draw annotations

## v0.2.4a1 (2022-10-08)

### Feat

- **base**: rename iters to base

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
- hooks.**init**
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
- move scheduler to top level
- schedulers
- enable dict ids
- visualize
- rename decorator mixin
- convert adapt dict to adapt list
- dict tensor
- generalized tensors
- detach context
- merge hooks.builder and hooks.modules
- rename multidetach to list detach
