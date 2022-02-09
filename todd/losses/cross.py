import einops
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..adapts import Decouple
from .base import BaseLoss
from .builder import LOSSES


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
