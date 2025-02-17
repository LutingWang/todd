__all__ = [
    'OpticalFlow',
    'SerializeMixin',
    'FloOpticalFlow',
    'Flo5OpticalFlow',
    'PfmOpticalFlow',
    'PngOpticalFlow',
]

import math
import pathlib
from abc import abstractmethod
from typing_extensions import Self

import cv2
import einops
import h5py
import numpy as np
import numpy.typing as npt
import torch

from todd.colors import ColorWheel

from .registries import OFEOpticalFlowRegistry


class OpticalFlow:

    def __init__(self, optical_flow: torch.Tensor) -> None:
        _, _, c = optical_flow.shape
        assert c == 2
        self._optical_flow = optical_flow

    @property
    def h(self) -> int:
        return self._optical_flow.shape[0]

    @property
    def w(self) -> int:
        return self._optical_flow.shape[1]

    @property
    def u(self) -> torch.Tensor:
        return self._optical_flow[:, :, 0]

    @property
    def v(self) -> torch.Tensor:
        return self._optical_flow[:, :, 1]

    @property
    def r(self) -> torch.Tensor:
        return (self.u**2 + self.v**2)**0.5

    @property
    def a(self) -> torch.Tensor:
        return torch.arctan2(-self.v, -self.u) / torch.pi

    def to_tensor(self) -> torch.Tensor:
        return self._optical_flow

    def to_color(
        self,
        color_wheel: ColorWheel | None = None,
        epsilon: float = 1e-5,
    ) -> torch.Tensor:
        if color_wheel is None:
            color_wheel = ColorWheel()
        i = (self.a + 1) / 2 * (len(color_wheel) - 1)
        r = self.r / (self.r.max() + epsilon)
        color = 255 - r.unsqueeze(-1) * (255 - color_wheel[i])
        return color.type(torch.uint8)


class SparseMixin(OpticalFlow):

    def __init__(
        self,
        optical_flow: torch.Tensor,
        validity: torch.Tensor,
    ) -> None:
        optical_flow[~validity] = 0
        super().__init__(optical_flow)
        self._validity = validity

    @property
    def validity(self) -> torch.Tensor:
        return self._validity


class SerializeMixin(OpticalFlow):
    SUFFIX: str

    @classmethod
    @abstractmethod
    def _load(cls, path: pathlib.Path) -> Self:
        pass

    @classmethod
    def load(cls, path: pathlib.Path) -> Self:
        assert path.suffix == cls.SUFFIX
        return cls._load(path)

    @abstractmethod
    def _dump(self, path: pathlib.Path) -> None:
        pass

    def dump(self, path: pathlib.Path) -> None:
        assert path.suffix == self.SUFFIX
        self._dump(path)


@OFEOpticalFlowRegistry.register_()
class FloOpticalFlow(SerializeMixin, OpticalFlow):
    SUFFIX = '.flo'
    MAGIC = 202021.25

    @classmethod
    def _load(cls, path: pathlib.Path) -> Self:
        with path.open('rb') as f:
            magic = np.fromfile(f, '<f', 1)
            assert magic == cls.MAGIC
            w = np.fromfile(f, '<i', 1).item()
            h = np.fromfile(f, '<i', 1).item()
            shape = (h, w, 2)
            data = np.fromfile(f, '<f', math.prod(shape)).reshape(shape)
        return cls(torch.tensor(data))

    def _dump(self, path: pathlib.Path) -> None:
        with path.open('wb') as f:
            np.array([self.MAGIC], '<f').tofile(f)
            np.array([self.w], '<i').tofile(f)
            np.array([self.h], '<i').tofile(f)
            data: npt.NDArray[np.float32] = self._optical_flow.numpy()
            data.astype('<f').tofile(f)


@OFEOpticalFlowRegistry.register_()
class Flo5OpticalFlow(SparseMixin, SerializeMixin, OpticalFlow):
    SUFFIX = '.flo5'

    @classmethod
    def _load(cls, path: pathlib.Path) -> Self:
        with h5py.File(path) as f:
            data = f['flow'][...]
        validity: npt.NDArray[np.bool_] = ~np.isnan(data)
        validity = validity.all(axis=-1)
        data[~validity] = 0
        return cls(torch.tensor(data).float(), torch.tensor(validity))

    def _dump(self, path: pathlib.Path) -> None:
        optical_flow = self._optical_flow.half().numpy()
        validity = self._validity.numpy()
        optical_flow[~validity] = np.nan
        with h5py.File(path, 'w') as f:
            f.create_dataset(
                'flow',
                data=optical_flow,
                compression='gzip',
                compression_opts=5,
            )


@OFEOpticalFlowRegistry.register_()
class PfmOpticalFlow(SerializeMixin, OpticalFlow):
    SUFFIX = '.pfm'
    HEADER = b'PF'

    @classmethod
    def _load(cls, path: pathlib.Path) -> Self:
        with path.open('rb') as f:
            header = f.readline().strip()
            assert header == cls.HEADER
            w, h = map(int, f.readline().split())
            scale = float(f.readline().strip())
            shape = (h, w, 3)
            data = np.fromfile(
                f,
                '<f' if scale < 0 else '>f',
                math.prod(shape),
            ).reshape(shape)
        data = data[:, :, :2]
        return cls(torch.tensor(data).flipud())

    def _dump(self, path: pathlib.Path) -> None:
        with path.open('wb') as f:
            f.write(self.HEADER + b'\n')
            f.write(f'{self.w} {self.h}\n'.encode())
            f.write(b'-1.0\n')
            data: npt.NDArray[np.float32] = np.zeros((self.h, self.w, 3), '<f')
            data[:, :, :2] = self._optical_flow.flipud().numpy()
            data.tofile(f)


@OFEOpticalFlowRegistry.register_()
class PngOpticalFlow(SerializeMixin, SparseMixin, OpticalFlow):
    SUFFIX = '.png'

    @classmethod
    def _load(cls, path: pathlib.Path) -> Self:
        data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        data = data.astype(np.int32)  # torch 2.0 does not support uint16
        tensor = torch.tensor(data)
        validity, tensor = tensor.split_with_sizes([1, 2], -1)
        validity = einops.rearrange(validity.bool(), 'h w 1 -> h w')
        tensor = tensor.float().flip(-1)
        tensor = (tensor - 2**15) / 64.
        return cls(tensor, validity)

    def _dump(self, path: pathlib.Path) -> None:
        tensor = self._optical_flow * 64 + 2**15
        tensor = tensor.flip(-1).int()
        validity = einops.rearrange(self._validity, 'h w -> h w 1').int()
        tensor = torch.cat([validity, tensor], -1)
        data: npt.NDArray[np.int32] = tensor.numpy()
        data = data.astype(np.uint16)
        cv2.imwrite(str(path), data)
