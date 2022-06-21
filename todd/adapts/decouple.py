import einops
import torch
import torch.nn as nn

from .base import ADAPTS, BaseAdapt


@ADAPTS.register_module()
class Decouple(BaseAdapt):

    def __init__(
        self,
        num: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num = num
        self._layer = nn.Linear(in_features, out_features * num, bias)

    def forward(self, feat: torch.Tensor, id_: torch.Tensor) -> torch.Tensor:
        """Decouple `feat`.

        Args:
            feat: n x dim
            pos: n

        Returns:
            decoupled_feat: n x dim
        """
        decoupled_feat: torch.Tensor = self._layer(feat)  # n x (num x dim)
        decoupled_feat = einops.rearrange(
            decoupled_feat,
            'n (num dim) -> n num dim',
            num=self._num,
        )
        decoupled_feat = decoupled_feat[  # yapf: disable
            torch.arange(id_.shape[0]), id_.long(),
        ]  # n x dim
        return decoupled_feat
