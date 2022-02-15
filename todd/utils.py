import numbers
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar
import warnings

import mmcv
from mmcv.runner import BaseModule, load_checkpoint
from mmcv.utils import Registry
import torch
import torch.nn as nn


_iter = None


def init_iter(iter_: int = 0):
    global _iter
    if _iter is not None:
        warnings.warn(f"iter={_iter} has been reset to {iter_}.")
    _iter = iter_


def get_iter() -> Optional[int]:
    return _iter


def inc_iter():
    global _iter
    _iter += 1


def getattr_recur(obj: Any, attr: str) -> Any:
    return eval('obj.' + attr)


def freeze_model(model: nn.Module):
    model.eval()
    model.requires_grad_(False)


def build(cls, cfg, **kwargs) -> Optional['BaseModule']:
    if cfg is None: return None
    module = cfg if isinstance(cfg, cls) else cls(cfg, **kwargs)
    return module


BaseModule.build = classmethod(build)


T = TypeVar('T', torch.Tensor, list, tuple, dict)


class CollectionTensor(Generic[T]):
    @staticmethod
    def apply(feat: T, op: Callable[[T], T]) -> T:
        if isinstance(feat, torch.Tensor):
            return op(feat)
        if isinstance(feat, list):
            return [CollectionTensor.apply(f, op) for f in feat]
        if isinstance(feat, tuple):
            return tuple(CollectionTensor.apply(f, op) for f in feat)
        if isinstance(feat, dict):
            return {k: CollectionTensor.apply(v, op) for k, v in feat.items()}
        raise TypeError(type(feat))

    @staticmethod
    def to(feat: T, device: torch.device) -> T:
        return CollectionTensor.apply(feat, lambda x: x.to(device))

    @staticmethod
    def cpu(feat: T) -> T:
        return CollectionTensor.apply(feat, lambda x: x.cpu())

    @staticmethod
    def cuda(feat: T) -> T:
        return CollectionTensor.apply(feat, lambda x: x.cuda())

    @staticmethod
    def allclose(feat1: T, feat2: T, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        if isinstance(feat1, torch.Tensor):
            assert isinstance(feat2, torch.Tensor)
            return torch.allclose(feat1, feat2)
        if isinstance(feat1, list) or isinstance(feat1, tuple):
            assert isinstance(feat2, list) or isinstance(feat2, tuple)
            assert len(feat1) == len(feat2)
            return all(CollectionTensor.allclose(f1, f2) for f1, f2 in zip(feat1, feat2))
        if isinstance(feat1, dict):
            assert isinstance(feat2, dict) and feat1.keys() == feat2.keys()
            return all(CollectionTensor.allclose(feat1[k], feat2[k]) for k in feat1)
        if isinstance(feat1, numbers.Number):
            assert isinstance(feat2, numbers.Number)
            return abs(feat1 - feat2) <= atol + rtol * abs(feat2)
        raise TypeError(type(feat1))


class ListTensor(CollectionTensor[T]):
    @staticmethod
    def stack(feat: T) -> torch.Tensor:
        if isinstance(feat, torch.Tensor):
            return feat
        feat = [ListTensor.stack(f) for f in feat]
        return torch.stack(feat)
    
    @staticmethod
    def shape(feat: T, depth: int = 0) -> Tuple[int]:
        if isinstance(feat, torch.Tensor):
            return feat.shape[max(depth, 0):]
        shape = {ListTensor.shape(f, depth - 1) for f in feat}
        assert len(shape) == 1
        shape = shape.pop()
        if depth <= 0:
            shape = (len(feat),) + shape
        return shape
    
    @staticmethod
    def new_empty(feat: T, *args, **kwargs) -> torch.Tensor:
        if isinstance(feat, torch.Tensor):
            return feat.new_empty(*args, **kwargs)
        return ListTensor.new_empty(feat[0], *args, **kwargs)
    
    @staticmethod
    def index(feat: T, pos: torch.Tensor) -> torch.Tensor:
        """Generalized `feat[pos]`.
    
        Args:
            feat: d_0 x d_1 x ... x d_(n-1) x *
            pos: m x n
    
        Returns:
            indexed_feat: m x *
        """
        m, n = pos.shape
        if isinstance(feat, torch.Tensor):
            assert n <= feat.ndim
        if m == 0:
            shape = ListTensor.shape(feat, n)
            return ListTensor.new_empty(feat, 0, *shape)
        if n == 0:
            feat = ListTensor.stack(feat)
            return feat.unsqueeze(0).repeat(m, *[1] * feat.ndim)
    
        pos = pos.long()
        if isinstance(feat, torch.Tensor):
            assert (pos >= 0).all()
            max_pos = pos.max(0).values
            feat_shape = pos.new_tensor(feat.shape)
            assert (max_pos < feat_shape[:n]).all(), f'max_pos({max_pos}) larger than feat_shape({feat_shape}).'
            return feat[pos.split(1, 1)].squeeze(1)
        indices = []
        indexed_feats = []
        for i, f in enumerate(feat):
            index = pos[:, 0] == i
            if not index.any():
                continue
            index, = torch.where(index)
            indexed_feat = ListTensor.index(f, pos[index, 1:])
            indices.append(index)
            indexed_feats.append(indexed_feat)
        indexed_feat = torch.cat(indexed_feats)
        index = torch.cat(indices)
        assert index.shape == pos.shape[:1]
        indexed_feat[index] = indexed_feat.clone()
        return indexed_feat


def clamp(x: torch.Tensor, min_: float = 0) -> torch.Tensor:
    if not x.is_cuda and x.dtype is torch.float16:
        x = x.float().clamp_min(min_).half()
    else:
        x = x.clamp_min(0)
    return x


def iou(bboxes1: torch.Tensor, bboxes2: Optional[torch.Tensor] = None, eps: float = 1e-6):
    """
    Args:
        bboxes1: *1 x 4
        bboxes2: *2 x 4
    
    Returns:
        ious: *1 x *2
    """
    flag = bboxes2 is None
    if flag:
        bboxes2 = bboxes1
    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return bboxes1.new_empty((bboxes1.shape[0], bboxes2.shape[0]))

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (
        area1 if flag else 
        (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    )

    lt = torch.maximum(  # [*1, *2, 2]
        bboxes1[:, None, :2],
        bboxes2[None, :, :2],
    )
    rb = torch.minimum(  # [*1, *2, 2]
        bboxes1[:, None, 2:],
        bboxes2[None, :, 2:],
    )

    wh = clamp(rb - lt)
    intersection = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - intersection
    union = clamp(union, eps)
    ious = intersection / union
    return ious


class ModelLoader:
    @staticmethod
    def load_mmlab_models(
        registry: Registry,
        config: str, 
        config_options: Optional[str] = None, 
        ckpt: Optional[str] = None,
    ) -> BaseModule:
        config_dict = mmcv.Config.fromfile(config)
        if config_options is not None:
            config_dict.merge_from_dict(config_options)
        model: BaseModule = registry.build(config_dict.model)
        if ckpt is not None:
            load_checkpoint(model, ckpt, map_location='cpu')
            model._is_init = True
        return model
