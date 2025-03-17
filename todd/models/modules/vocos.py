__all__ = [
    'Vocos',
]

from abc import ABC, abstractmethod

import einops
import torch
import torchaudio
from torch import nn

from ...utils import StateDict, StateDictConverter
from ...utils.state_dicts import SequentialStateDictConverterMixin
from .pretrained import PretrainedMixin
from .transformer import mlp


class BaseConvNeXt(PretrainedMixin, ABC):

    def __init__(self, *args, channels: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._channels = channels
        self._in_conv = nn.Conv1d(
            channels,
            channels,
            7,
            padding=3,
            groups=channels,
        )
        self._norm = nn.LayerNorm(channels, eps=1e-6)

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, 'b t c -> b c t')
        x = self._in_conv(x)
        x = einops.rearrange(x, 'b c t -> b t c')
        x = self._norm(x)
        return x

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self._input(x)
        x = self._forward(x)
        return identity + x


class ConvNeXtStateDictConverter(
    SequentialStateDictConverterMixin,
    StateDictConverter,
):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'dwconv\.(.*)', r'_in_conv.\1')
        self._register_regex_converter(r'norm\..*', r'_\g<0>')
        self._register_regex_converter(r'pwconv1\.(.*)', r'_mlp.0.\1')
        self._register_regex_converter(r'pwconv2\.(.*)', r'_mlp.2.\1')
        self._register_key_mapping('gamma', '_gamma')


class ConvNeXt(BaseConvNeXt):
    STATE_DICT_CONVERTER = ConvNeXtStateDictConverter

    def __init__(
        self,
        *args,
        hidden_channels: int = 1536,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._mlp = mlp(self._channels, hidden_channels, self._channels)
        self._gamma = nn.Parameter(torch.empty(self._channels))

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._mlp(x)
        x = self._gamma * x
        return x


class VocosStateDictConverter(StateDictConverter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_regex_converter(r'embed\.(.*)', r'_in_conv.\1')
        self._register_regex_converter(r'norm\.(.*)', r'_in_norm.\1')
        self._register_child_converter(
            'convnext',
            '_blocks',
            ConvNeXtStateDictConverter,
        )
        self._register_regex_converter(
            r'final_layer_norm\.(.*)',
            r'_out_norm.\1',
        )
        self._register_regex_converter(r'head\.out\.(.*)', r'_projector.\1')

        self._register_regex_converter(r'feature_extractor\..*', None)
        self._register_regex_converter(r'head\.istft\..*', None)

    def _pre_convert(self, state_dict: StateDict) -> StateDict:
        state_dict = super()._pre_convert(state_dict)
        state_dict = {
            k.removeprefix('backbone.'): v
            for k, v in state_dict.items()
        }
        return state_dict


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

        blocks = [ConvNeXt(channels=hidden_channels) for _ in range(depth)]
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
        x = self._blocks(x)
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
