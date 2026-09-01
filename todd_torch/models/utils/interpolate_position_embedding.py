__all__ = [
    'interpolate_position_embedding',
]

import einops
import torch
import torch.nn.functional as F


def interpolate_position_embedding(
    position_embedding: torch.Tensor,
    wh: tuple[int, int],
    new_wh: tuple[int, int],
    offset: float | None = None,
    **kwargs,
) -> torch.Tensor:
    if wh == new_wh:
        return position_embedding

    w, h = wh
    new_w, new_h = new_wh

    if position_embedding.shape[0] == w * h:
        cls_embedding = None
    else:
        cls_embedding = position_embedding[[0]]
        position_embedding = position_embedding[1:]

    position_embedding = einops.rearrange(
        position_embedding,
        '(h w) c -> 1 c h w',
        h=h,
        w=w,
    )

    if offset is None:
        position_embedding = F.interpolate(
            position_embedding,
            (new_h, new_w),
            **kwargs,
        )
    else:
        position_embedding = F.interpolate(
            position_embedding,
            scale_factor=((new_h + offset) / h, (new_w + offset) / w),
            **kwargs,
        )

    position_embedding = einops.rearrange(
        position_embedding,
        '1 c h w -> (h w) c',
    )

    if cls_embedding is not None:
        position_embedding = torch.cat([cls_embedding, position_embedding])

    return position_embedding
