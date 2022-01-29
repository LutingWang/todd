from functools import cached_property, reduce
from typing import Dict, List, Tuple, Union

import einops
import torch

from .base import BaseAdapt
from .builder import ADAPTS


class DictTensor:
    def __init__(self, keys: Union[torch.Tensor, List[Tuple[int]]], values: torch.Tensor):
        if isinstance(keys, torch.Tensor):
            keys = [tuple(key) for key in keys.int().tolist()]
        self._keys: List[Tuple[int]] = keys
        self._values = values

    def __getitem__(self, keys: List[Tuple[int]]) -> torch.Tensor:
        inds = self.inds(keys)
        return self._values[inds]

    @property
    def keys(self) -> List[Tuple[int]]:
        return self._keys

    @property
    def values(self) -> torch.Tensor:
        return self._values

    @cached_property
    def key2ind(self) -> Dict[Tuple[int], int]:
        return {key: i for i, key in enumerate(self._keys)}

    def inds(self, keys: List[Tuple[int]]) -> List[int]:
        return [self.key2ind[key] for key in keys]


def _union(dict_tensors: List[DictTensor]) -> Tuple[List[DictTensor], DictTensor]:
    key_set = list(set(pos for dict_tensor in dict_tensors for pos in dict_tensor.keys))
    n = len(key_set)
    s = len(dict_tensors)

    mask = DictTensor(key_set, torch.zeros((n, s)))
    union_dict_tensors = []
    for i, dict_tensor in enumerate(dict_tensors):
        inds = mask.inds(dict_tensor.keys)
        mask[inds][i] = 1

        shape = (n,) + dict_tensor.values.shape[1:]
        union_dict_tensor = DictTensor(key_set, dict_tensor.values.new_zeros(shape))
        union_dict_tensor[inds] = dict_tensor.values
        union_dict_tensors.append(union_dict_tensor)
    return union_dict_tensors, mask


def _intersect(dict_tensors: List[DictTensor]) -> List[DictTensor]:
    key_set = list(reduce(lambda a, b: a & b, (set(dict_tensor.keys) for dict_tensor in dict_tensors)))
    intersect_dict_tensors = [DictTensor(key_set, dict_tensor[key_set]) for dict_tensor in dict_tensors]
    return intersect_dict_tensors


@ADAPTS.register_module()
class Union(BaseAdapt):
    def forward(self, feats: List[torch.Tensor], ids: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match `feats` accroding to their `poses`.
    
        Align the `feats` coming from different sources to have same `matched_pos` and stack them togethor. 
        For positions where some of `feats` do not show up, an all-zero tensor is added as default. 
        A 2D `mask` is returned to indicate the type of a matched feature, where `1` corresponds to features coming from `feats` and `0` for added default all-zero tensors.
    
        Args:
            feats: [n_s x d_1 x d_2 x ... x d_m]
                Features from `s` different sources, each source can have different `n_s`.
            ids: [n_s x m]
                Positions of each feature.
        
        Returns:
            union_feats: s x n x d_1 x d_2 x ... x d_m
            ids: n x m
            mask: s x n
        """
        dict_tensors = [DictTensor(id_, feat) for feat, id_ in zip(feats, ids)]
        union_dict_tensors, mask = _union(dict_tensors)
        union_feats = torch.stack([dict_tensor.values] for dict_tensor in union_dict_tensors)
        ids = torch.Tensor(mask.keys)
        mask = einops.rearrange(mask.values, 'n s -> s n')
        return union_feats, ids, mask


@ADAPTS.register_module()
class Intersect(BaseAdapt):
    def forward(self, feats: List[torch.Tensor], ids: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match positions that show up both in `pred_poses` and `target_poses`.

        Args:
            feats: [n_s x d_1 x d_2 x ... x d_m]
                Features from `s` different sources, each source can have different `n_s`.
            ids: [n_s x m]
                Positions of each feature.

        Returns:
            intersect_feats: [n x d_1 x d_2 x ... x d_m]
        """
        dict_tensors = [DictTensor(id_, feat) for feat, id_ in zip(feats, ids)]
        intersect_dict_tensors = _intersect(dict_tensors)
        intersect_feats = [dict_tensor.values for dict_tensor in intersect_dict_tensors]
        return intersect_feats
