__all__ = [
    'sinusoidal_position_embedding',
    'rotary_position_embedding',
]

import einops
import torch


def sinusoidal_position_embedding(x: torch.Tensor, d: int) -> torch.Tensor:
    assert d % 2 == 0
    scaled_x: torch.Tensor = (
        einops.rearrange(x, '... -> ... 1')
        / 10000**torch.linspace(0, 1, d // 2, device=x.device)
    )
    position_embedding = torch.stack((scaled_x.sin(), scaled_x.cos()), -1)
    position_embedding = einops.rearrange(
        position_embedding,
        '... d two -> ... (d two)',
        two=2,
    )
    return position_embedding


def rotary_position_embedding(
    x: torch.Tensor,
    position_embedding: torch.Tensor,
) -> torch.Tensor:
    d = position_embedding.shape[-1]
    identity = x[..., d:]
    x = x[..., :d]

    x1, x2 = einops.rearrange(x, '... (d two) -> ... d two', two=2).unbind(-1)
    sin, cos = einops.rearrange(
        position_embedding,
        '... (d two) -> ... d two',
        two=2,
    ).unbind(-1)
    x = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1)
    x = einops.rearrange(x, '... d two -> ... (d two)')

    return torch.cat((x, identity), -1)
