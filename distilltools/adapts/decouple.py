from typing import List, Optional, Tuple

import einops
import torch
import torch.nn as nn

from .base import BaseAdapt
from .builder import ADAPTS


def match_by_poses(feats: List[torch.Tensor], poses: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match `feats` accroding to their `poses`.

    Align the `feats` coming from different sources to have same `matched_pos` and stack them togethor. 
    For positions where some of `feats` do not show up, an all-zero tensor is added as default. 
    A 2D `mask` is returned to indicate the type of a matched feature, where `1` corresponds to features coming from `feats` and `0` for added default all-zero tensors.

    Args:
        feats: [n_s x dim]
            Features from `s` different sources, each source can have different `n_s`.
        poses: [n_s x 5]
            Positions of each feature.
    
    Returns:
        matched_feats: n x s x dim
        matched_pos: n x 5
        mask: n x s
    """
    pos_set = list(set(tuple(pos) for pos in torch.cat(poses).int().tolist()))
    n = len(pos_set)
    s = len(feats)
    dim = feats[0].shape[1]
    matched_feats = feats[0].new_zeros((n, s, dim))
    matched_poses = torch.Tensor(pos_set)
    mask = feats[0].new_zeros((n, s))

    pos2ind = {pos: i for i, pos in enumerate(pos_set)}
    for i, (target, pos) in enumerate(zip(feats, poses)):
        inds = [pos2ind[tuple(p)] for p in pos.int().tolist()]
        matched_feats[inds, i] = target
        mask[inds, i] = 1
    return matched_feats, matched_poses, mask


def match_poses(pred_poses: torch.Tensor, target_poses: torch.Tensor) -> Tuple[List[int], List[int]]:
    """Match positions that show up both in `pred_poses` and `target_poses`.

    Args:
        pred_poses: n x 5
        target_poses: m x 5

    Returns:
        pred_inds: n'
        target_inds: n'
    """
    pred_poses = [tuple(pos) for pos in pred_poses.int().tolist()]
    target_poses = [tuple(pos) for pos in target_poses.int().tolist()]
    pos_dict = {
        pos: i for i, pos in enumerate(pred_poses)
    }
    match_result = [
        (pos_dict[pos], j) for j, pos in enumerate(target_poses) 
        if pos in pos_dict
    ]
    if len(match_result) == 0:
        return None, None
    pred_inds, target_inds = zip(*match_result)
    return list(pred_inds), list(target_inds)


def index_by_pos(feats: List[torch.Tensor], pos: torch.Tensor, stride: int = 1) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Index feat from `feats` accroding to `pos`.

    For each valid position in `pos`, retrieve the corresponding feat in `feats`, and organize them in a 2D matrix. 
    Invalid position in `pos` represents gt, wich will be skipped during indexing.
    As a result, `indexed_feats` will not maintain their sequence in pos. Use the returned `indices` to recover sequence.

    Args:
        feats: [bs x dim x h_l x w_l]
            Features from each of the `l` level.
        pos: n x 5
            Index in sequence of `[bs, l, h, w, anchor]`.
        stride:
            Stride of `feats` relative to `pos`. E.g. when `stride` is 2, position (96, 100) and (97, 101) both refer to (48. 50) of `feats`.
    
    Returns:
        indices: n'
            `feats[i]` comes from `pos[inds[i]]` of `feats`.
        indexed_feats: n' x dim
    """
    indices = []
    indexed_feats = []
    for level, feat in enumerate(feats):
        inds, = torch.where(pos[:, 1] == level)
        if inds.numel() == 0: continue
        bs, _, h, w, _ = pos[inds].long().t()
        h = h // stride
        w = w // stride
        assert h.max() <= feat.shape[2] and w.max() <= feat.shape[3], f"Position ({h.max()}, {w.max()}) is invalid for feature map with size {feat.shape}."
        h = h.clamp(None, feat.shape[2] - 1)
        w = w.clamp(None, feat.shape[3] - 1)
        indices.append(inds)
        indexed_feats.append(feat[bs, :, h, w])
    if indices == []:
        return None, None
    else: 
        return torch.cat(indices), torch.cat(indexed_feats)


@ADAPTS.register_module()
class Decouple(BaseAdapt):
    def __init__(self, in_features: int, out_features: int, num: int = 1, bias: bool = ..., **kwargs):
        super().__init__(**kwargs)
        self._num = num
        self._layer = nn.Linear(in_features, out_features * num, bias)

    def forward(self, feats: List[torch.Tensor], pos: torch.Tensor) -> torch.Tensor:
        """Decouple `feats`.

        Args:
            feats: [n x dim x h x w]
            pos: m x 5

        Returns:
            decoupled_feats: n x dim
        """
        indices, indexed_feats = index_by_pos(feats, pos)
        if indices is None: return None
        pos = pos[indices]
        indexed_feats: torch.Tensor = self._layer(indexed_feats)  # n x (num x dim)
        if self._num > 1:
            indexed_feats = einops.rearrange(indexed_feats, 'n (num dim) -> n num dim', num=self._num)
            indexed_feats = indexed_feats[torch.arange(indexed_feats.shape[0]), pos[:, -1].long()]  # n x dim
        return indexed_feats
