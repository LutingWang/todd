__all__ = [
    'Vocos',
]

import einops
import torch
import torchaudio
from torch import nn

from ...utils import StateDictConverter
from .pretrained import PretrainedMixin
from .transformer import mlp

# TODO: refactor state dict converter


class VocosStateDictConverter(StateDictConverter):

    def _convert_backbone_convnext(self, key: str) -> str:
        index = key.index('.') + 1
        prefix = key[:index]
        key = key[index:]

        if key.startswith('dwconv.'):
            key = key.removeprefix('dwconv.')
            key = '_in_conv.' + key
        elif key.startswith(('norm.', 'gamma')):
            key = '_' + key
        elif key.startswith('pwconv1.'):
            key = key.removeprefix('pwconv1.')
            key = '_mlp.0.' + key
        elif key.startswith('pwconv2.'):
            key = key.removeprefix('pwconv2.')
            key = '_mlp.2.' + key
        else:
            raise ValueError(f'Unknown key: {key}')

        return '_blocks.' + prefix + key

    def _convert_backbone(self, key: str) -> str:
        # return f'backbone.{key}'
        if key.startswith('convnext.'):
            key = key.removeprefix('convnext.')
            return self._convert_backbone_convnext(key)
        if key.startswith('embed.'):
            key = key.removeprefix('embed.')
            return f'_in_conv.{key}'
        if key.startswith('norm.'):
            return f'_in_{key}'
        if key.startswith('final_layer_norm.'):
            key = key.removeprefix('final_layer_norm.')
            return f'_out_norm.{key}'
        raise ValueError(f'Unknown key: {key}')

    def _convert_head(self, key: str) -> str:
        assert key.startswith('out.')
        key = key.removeprefix('out.')
        return f'_projector.{key}'

    def _convert(self, key: str) -> str | None:
        if key.startswith('feature_extractor.'):
            return None
        if key.startswith('backbone.'):
            key = key.removeprefix('backbone.')
            return self._convert_backbone(key)
        if key.startswith('head.'):
            key = key.removeprefix('head.')
            if key.startswith('istft.'):
                return None
            return self._convert_head(key)
        return key


class ConvNeXtBlock(nn.Module):

    def __init__(
        self,
        *args,
        channels: int,
        hidden_channels: int = 1536,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._in_conv = nn.Conv1d(
            channels,
            channels,
            7,
            padding=3,
            groups=channels,
        )
        self._norm = nn.LayerNorm(channels, 1e-6)
        self._mlp = mlp(channels, hidden_channels, channels)
        self._gamma = nn.Parameter(torch.empty(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self._in_conv(x)
        x = einops.rearrange(x, 'b c t -> b t c')
        x = self._norm(x)
        x = self._mlp(x)
        x = self._gamma * x
        x = einops.rearrange(x, 'b t c -> b c t')
        return identity + x


class Vocos(PretrainedMixin):
    STATE_DICT_CONVERTER = VocosStateDictConverter

    def __init__(
        self,
        *args,
        sample_rate: int,
        n_fft: int = 1024,
        hop_length: int | None = None,
        mel_channels: int = 100,
        hidden_channels: int = 512,
        depth: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if hop_length is None:
            hop_length = n_fft // 4

        self._mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate,
            n_fft,
            hop_length=hop_length,
            n_mels=mel_channels,
            power=1,
        )

        self._in_conv = nn.Conv1d(mel_channels, hidden_channels, 7, padding=3)
        self._in_norm = nn.LayerNorm(hidden_channels, 1e-6)

        blocks = [
            ConvNeXtBlock(channels=hidden_channels) for _ in range(depth)
        ]
        self._blocks = nn.Sequential(*blocks)

        self._out_norm = nn.LayerNorm(hidden_channels, 1e-6)
        self._projector = nn.Linear(hidden_channels, n_fft + 2)

        self.window = torch.hann_window(n_fft)

    @property
    def sample_rate(self) -> int:
        return self._mel_spectrogram.sample_rate

    @property
    def n_fft(self) -> int:
        return self._mel_spectrogram.n_fft

    @property
    def hop_length(self) -> int:
        return self._mel_spectrogram.hop_length

    @property
    def mel_channels(self) -> int:
        return self._mel_spectrogram.n_mels

    @property
    def window(self) -> torch.Tensor:
        return self.get_buffer('_window')

    @window.setter
    def window(self, value: torch.Tensor) -> None:
        self.register_buffer('_window', value, False)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self._mel_spectrogram(audio)

    def decode(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self._in_conv(mel_spectrogram)
        x = einops.rearrange(x, 'b c t -> b t c')
        x = self._in_norm(x)
        x = einops.rearrange(x, 'b t c -> b c t')
        x = self._blocks(x)
        x = einops.rearrange(x, 'b c t -> b t c')
        x = self._out_norm(x)
        x = self._projector(x)
        x = einops.rearrange(x, 'b t c -> b c t')
        magnitude, phase = x.chunk(2, 1)
        magnitude = magnitude.exp().clip(max=1e2)
        return torch.istft(  # pylint: disable=not-callable
            magnitude * (phase.cos() + 1j * phase.sin()),
            self.n_fft,
            self.hop_length,
            window=self.window,
        )
