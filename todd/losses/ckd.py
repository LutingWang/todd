from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..utils import iou

from .base import BaseLoss
from .builder import LOSSES


def ckd_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    ignore: Tuple[torch.Tensor, torch.Tensor], 
    gamma: float = 0.07,
) -> torch.Tensor:
    """Warpper of CKD loss.

    Refer to http://arxiv.org/abs/2108.07482.
    
    Args:
        pred: n x dim
            Normalized predictions.
        target: m x dim
            Normalized targets. First `n` targets correspond to `pred`.
        ignore: (n, n)

    Returns:
        loss: 1
    """
    similarity = target.mm(pred.t()) / gamma  # m x n
    similarity = similarity.exp()
    pos = torch.diag(similarity)  # n
    similarity[ignore] = 0
    total = similarity.sum(0)  # n
    loss = pos / total  # n
    return -loss.log()


class MemoryPool:
    def __init__(self, size: int = 10):
        self._memory = []
        self._size = size
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()

    def register(self, tensor: torch.Tensor) -> int:
        tensor = tensor.contiguous()
        tensor_list = [torch.zeros_like(tensor) for _ in range(self._world_size)]
        dist.all_gather(tensor_list, tensor)
        tensor_list[0], tensor_list[self._rank] = tensor_list[self._rank], tensor_list[0]
        gathered_tensor = torch.cat(tensor_list)
        self._memory.insert(0, gathered_tensor)
        if len(self._memory) > self._size:
            self._memory.pop(-1)
    
    @property
    def memory(self) -> torch.Tensor:
        return torch.cat(self._memory)


@LOSSES.register_module()
class CKDLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory_pool = MemoryPool()

    def forward(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor, 
        bboxes: List[torch.Tensor],
    ):
        """Compute CKD loss.

        Refer to http://arxiv.org/abs/2108.07482.
    
        Args:
            preds: m x dim
            targets: m x dim
            bboxes: n x m x 4

        Returns:
            loss: 1
        """
        assert preds.shape == targets.shape, (preds.shape, targets.shape)

        ignore, ind = [], 0
        for bbox in bboxes:
            ious = iou(bbox)
            x, y = torch.where(ious > 0.5)
            x += ind
            y += ind
            ind += bbox.shape[0]
            ignore.append((x, y))
        assert ind == preds.shape[0], (ind, preds.shape)
        x, y = zip(*ignore)
        ignore = (torch.cat(x), torch.cat(y))

        preds = F.normalize(preds)
        targets = F.normalize(targets)
        self._memory_pool.register(targets)
        loss = ckd_loss(preds, self._memory_pool.memory, ignore)
        return super().forward(loss)
