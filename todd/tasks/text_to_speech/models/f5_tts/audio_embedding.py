__all__ = [
    'AudioEmbedding',
]

import einops
import torch
from torch import nn


class AudioEmbedding(nn.Module):

    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        duration: int,
    ) -> torch.Tensor:
        b, c, t = mel_spectrogram.shape
        embedding = mel_spectrogram.new_zeros(b, c, duration)
        embedding[..., :t] = mel_spectrogram.clamp_min(1e-5).log()
        embedding = einops.rearrange(embedding, 'b c t -> b t c')
        return embedding
