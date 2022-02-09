import einops
from mmcv.cnn import ConvModule
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSSES
from .functional import MSELoss


@LOSSES.register_module()
class SGFILoss(MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed = nn.Sequential(
            ConvModule(256, 128, 3, stride=2),
            ConvModule(128, 64, 3, stride=2)
        )
        self.tau = nn.Parameter(torch.FloatTensor(data=[1]))

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, 
        pred_poses: torch.Tensor = None, target_poses: torch.Tensor = None, 
        *args, **kwargs,
    ):
        """Re-implementation of G-Det.

        Refer to http://arxiv.org/abs/2108.07482.

        Args:
            pred: (l x r) x c x h x w
                Each of the `l` levels generate `r` RoIs. Typical shape is (4 x 1024) x 256 x 7 x 7.
            target: r x c x h x w
            pred_poses: r x 5
            target_poses: r x 5

        Returns:
            loss: 1
        """
        assert (pred_poses is not None) == (target_poses is not None)

        # If need to `match_poses`, use `rearranged_pred` to update `pred`
        rearranged_pred = einops.rearrange(pred, '(level roi) c h w -> roi level c h w', level=4)
        if pred_poses is not None:
            pred_inds, target_inds = match_poses(pred_poses, target_poses)
            rearranged_pred = rearranged_pred[pred_inds]  # r x l x c x h x w
            pred = einops.rearrange(rearranged_pred, 'roi level c h w -> (level roi) c h w')
            target = target[target_inds]

        embed_pred: torch.Tensor = self.embed(pred)  # (l x r) x 64 x 1 x 1
        embed_pred = einops.rearrange(embed_pred, '(level roi) c 1 1 -> roi level c', level=4)  # c == 64
        embed_target = self.embed(target)  # r x 64 x 1 x 1
        embed_target = embed_target.squeeze(-1)  # r x 64 x 1
        similarity: torch.Tensor = embed_pred.bmm(embed_target)  # r x l x 1
        similarity = F.softmax(similarity / self.tau, dim=1)
        similarity = einops.rearrange(similarity, 'roi level 1 -> roi level 1 1 1')

        fused_pred = (rearranged_pred * similarity).sum(1)  # r x c x h x w
        return super().forward(fused_pred, target, *args, **kwargs)
