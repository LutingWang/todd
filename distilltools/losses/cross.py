from typing import List, Tuple

import einops
from einops.layers.torch import Rearrange
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..adapts import Decouple
from .base import BaseLoss
from .builder import LOSSES
from .mse import MSELoss
from .utils import index_by_pos, match_by_poses, weight_loss


@LOSSES.register_module()
class CrossLoss(MSELoss):
    def __init__(self, decouple_by_id: bool = True, stride: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decouple_by_id = decouple_by_id
        self.stride = stride
        self.init_decouple_layer()
    
    def init_decouple_layer(self):
        self.decouple_layer = Decouple(256, 1024, 9 if self.decouple_by_id else 1)

    def forward_dummy(self, feat: torch.Tensor) -> torch.Tensor:
        """Forward using all parameters.

        Args:
            feat: 1 x dim
        
        Returns:
            loss: 1
        """
        feat: torch.Tensor = self.decouple_layer(feat, None)
        return feat.mean() * 0

    def match(self, feat: List[torch.Tensor], poses: torch.Tensor, *args: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Index and match feat and target.

        Args:
            feat: [n x dim x h x w]
            poses: m x 5
            args: [m x ...]
                Other tensors need to be matched.

        Returns:
            indexed_feats: n' x num x dim | 1
                When all `poses` are invalid, return 0.
            args: [n' x ...]
        """
        assert len(args) > 0
        indices, indexed_feats = index_by_pos(feat, poses, self.stride)  # n, n x dim
        if indices is None:
            return self.forward_dummy(feat[0][[0], :, [0], [0]]), [None] * len(args)
        poses = poses[indices]

        indexed_feats = self.decouple_layer(
            indexed_feats, poses[:, -1] if self.decouple_by_id else None,
        )  # n' x 1 x dim | n' x num x dim
        args = [tensor[indices] for tensor in args]
        return indexed_feats, args

    def forward(self, feat: List[torch.Tensor], target: torch.Tensor, poses: torch.Tensor):
        feat, (target,) = self.match(feat, poses, target)
        if feat.numel() <= 1:
            return feat
        feat = einops.rearrange(feat, 'n 1 dim -> n dim')
        return super().forward(feat, target)


@LOSSES.register_module()
class DoubleHeadCrossLoss(CrossLoss):
    def init_decouple_layer(self):
        self.cls_decouple_layer = Decouple(256, 1024, 9 if self.decouple_by_id else 1)
        self.reg_decouple_layer = Decouple(256, 1024, 9 if self.decouple_by_id else 1)
        if self.decouple_by_id:
            self.decouple_layer = lambda x, id_: (
                self.cls_decouple_layer(x[..., :256], id_), 
                self.reg_decouple_layer(x[..., 256:], id_),
            )
        else:
            self.decouple_layer = lambda x: (
                self.cls_decouple_layer(x[..., :256]), 
                self.reg_decouple_layer(x[..., 256:]),
            )

    def forward_dummy(self, feat: torch.Tensor) -> torch.Tensor:
        feat: torch.Tensor = self.decouple_layer(feat)
        feat = sum(f.mean() * 0 for f in feat)
        return feat

    def forward(self, cls_feat: torch.Tensor, reg_feat: torch.Tensor, cls_target: torch.Tensor, reg_target: torch.Tensor, poses: torch.Tensor):
        reg_target = einops.rearrange(reg_target, 'n dim 1 1 -> n dim')
        cls_reg_feat = [torch.cat((c, r), dim=1) for c, r in zip(cls_feat, reg_feat)]
        cls_reg_feat, (cls_target, reg_target) = self.match(cls_reg_feat, poses, cls_target, reg_target)
        if not isinstance(cls_reg_feat, tuple):
            return cls_reg_feat
        cls_feat, reg_feat = cls_reg_feat
        cls_feat = einops.rearrange(cls_feat, 'n 1 dim -> n dim')
        reg_feat = einops.rearrange(reg_feat, 'n 1 dim -> n dim')
        return {
            'cls': MSELoss.forward(self, cls_feat, cls_target), 
            'reg': MSELoss.forward(self, reg_feat, reg_target),
        }
        

@LOSSES.register_module()
class FuseCrossLoss(CrossLoss):
    def init_decouple_layer(self):
        self.decouple_layer = Decouple(512, 1024, 9 if self.decouple_by_id else 1)

    def match_fuse(self, cls_feat: torch.Tensor, reg_feat: torch.Tensor, poses: torch.Tensor, *args: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        cls_reg_feat = [torch.cat((c, r), dim=1) for c, r in zip(cls_feat, reg_feat)]
        return super().match(cls_reg_feat, poses, *args)

    def forward(self, cls_feat: torch.Tensor, reg_feat: torch.Tensor, target: torch.Tensor, poses: torch.Tensor):
        cls_reg_feat = [torch.cat((c, r), dim=1) for c, r in zip(cls_feat, reg_feat)]
        return super().forward(cls_reg_feat, target, poses)


@LOSSES.register_module()
class MultiTeacherCrossLoss(CrossLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed = nn.Sequential(
            Rearrange('n s dim -> (n s) dim'),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.tau = nn.Parameter(torch.FloatTensor(data=[1]))

    def forward_dummy(self, feat: torch.Tensor) -> torch.Tensor:
        """Forward using all parameters.

        Args:
            feat: 1 x dim
        
        Returns:
            loss: 1
        """
        loss = super().forward_dummy(feat)
        feat = self.embed(feat) / self.tau
        return loss + feat.mean() * 0

    def forward(self, feat: torch.Tensor, *args):
        targets: List[torch.Tensor] = args[::2]
        poses: List[torch.Tensor] = args[1::2]
        num_targets = len(targets)
        assert len(poses) == num_targets > 1, f"Number of targets ({num_targets}) and poses ({len(poses)}) should be same and greater than 1."

        targets, poses, mask = match_by_poses(targets, poses)  # n x s x dim, n x 5, n x s
        
        feats, (targets, mask) = self.match(feat, poses, targets, mask)
        if feats.numel() <= 1:
            return feats

        embed_feats: torch.Tensor = self.embed(feats)  # (n x s) x dim
        embed_feats = einops.rearrange(embed_feats, '(n s) dim -> n s dim', s=1)
        embed_targets: torch.Tensor = self.embed(targets)  # (n x s) x dim
        embed_targets = einops.rearrange(embed_targets, '(n s) dim -> n s dim', s=num_targets)
        similarity = embed_feats * embed_targets
        similarity = similarity.sum(-1, keepdim=True) / self.tau  # n x s x 1
        similarity[~mask.bool()] = -float('inf')
        similarity = similarity.softmax(1)  # n x s x 1

        feats = einops.rearrange(feats, 'n 1 dim -> n dim')
        targets: torch.Tensor = similarity * targets  # n x s x dim
        targets = targets.sum(1)  # n x dim
        return MSELoss.forward(self, feats, targets)
        # return MSELoss.forward(self, feats, targets, weight=similarity)


@weight_loss
def ckd_loss(pred: torch.Tensor, target: torch.Tensor, overlaps: torch.Tensor, offset: int, gamma: float = 0.07):
    """Warpper of CKD loss.

    Refer to http://arxiv.org/abs/2108.07482.
    
    Args:
        pred: n x dim
            Normalized predictions.
        target: m x dim
            Normalized targets. First `n` targets counting from `offset` are corresponding to `pred`.
        overlaps: n x n
            Overlaps between `pred` and its corresponding `target`.

    Returns:
        loss: 1
    """
    similarity = target.mm(pred.t()) / gamma  # m x n
    similarity = similarity.exp()
    mask = torch.ones_like(similarity, dtype=torch.bool)
    mask[offset:offset + pred.shape[0]] = overlaps < 0.5
    similarity = similarity * mask

    pos = torch.diag(similarity, -offset)
    total = similarity.sum(0)  # n
    loss = pos / (total - pos)  # n
    return -loss.log()


class MemoryPool:
    def __init__(self, size: int = 10):
        self._memory = []
        self.size = size
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def register(self, tensor: torch.Tensor) -> int:
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        gathered_tensor = torch.cat(tensor_list)
        self._memory.insert(0, gathered_tensor)
        if len(self._memory) > self.size:
            self._memory.pop(-1)
        return self.rank * tensor.shape[0]
    
    @property
    def memory(self) -> torch.Tensor:
        return torch.cat(self._memory)


@LOSSES.register_module()
class CKDLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diag_mask = ~torch.diag(torch.ones(1024)).bool().cuda()
        self.decouple_layer = Decouple(256, 1024, 9, False)
        self.memory_pool = MemoryPool()

    @staticmethod
    def _iou(bboxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        assert bboxes.shape[0] > 0
        lt = torch.max(bboxes[None, :, :2], bboxes[:, None, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes[None, :, 2:], bboxes[:, None, 2:])  # [rows, cols, 2]
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]

        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        union = area[:, None] + area[None, :] - overlap
        ious = overlap / union.clamp(min=eps)
        return ious

    def forward_dummy(self, feat: torch.Tensor) -> torch.Tensor:
        """Forward using all parameters.

        Args:
            feat: 1 x dim
        
        Returns:
            loss: 1
        """
        loss: torch.Tensor = self.decouple_layer(feat)
        return loss.mean() * 0

    def forward(self, pred: torch.Tensor, target: torch.Tensor, pos: torch.Tensor, bboxes: torch.Tensor):
        """Compute CKD loss.

        Refer to http://arxiv.org/abs/2108.07482.
    
        Args:
            pred: [bs x dim x h x w]
                Multilevel classification features.
            target: m x dim
                Normalized targets. First `n` targets are corresponding to `pred`.
            pos: m x 5
            bboxes: m x 4

        Returns:
            loss: 1
        """
        batch: torch.Tensor = einops.repeat(pos[:, 0], 'n -> n repeat', repeat=pos.shape[0])
        batch_mask = batch == batch.T
        diag_mask = self.diag_mask[:pos.shape[0], :pos.shape[0]]
        overlaps = self._iou(bboxes) * batch_mask * diag_mask

        indices, indexed_preds = index_by_pos(pred, pos)
        if indices is None:
            return self.forward_dummy(pred[0][[0], :, [0], [0]])
        pos = pos[indices]
        overlaps = overlaps[indices][:, indices]

        indexed_preds = self.decouple_layer(indexed_preds, pos[:, -1])  # n x 1 x dim
        indexed_preds = einops.rearrange(indexed_preds, 'n 1 dim -> n dim')
        indexed_preds = F.normalize(indexed_preds)

        target = target[indices]
        target = F.normalize(target)
        offset = self.memory_pool.register(target.contiguous())
        loss = ckd_loss(indexed_preds, self.memory_pool.memory, overlaps, offset)
        return super().forward(loss)
