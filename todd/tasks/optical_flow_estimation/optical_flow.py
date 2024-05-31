__all__ = [
    'OpticalFlow',
    'SerializableOpticalFlow',
    'FloOpticalFlow',
    'PfmOpticalFlow',
    'PngOpticalFlow',
]

import math
import pathlib
from abc import abstractmethod
from typing import Container
from typing_extensions import Self

import cv2
import einops
import numpy as np
import numpy.typing as npt
import torch

from ...colors import ColorWheel


class OpticalFlow:

    def __init__(
        self,
        optical_flow: torch.Tensor,
        validity: torch.Tensor | None = None,
    ) -> None:
        _, _, c = optical_flow.shape
        assert c == 2
        optical_flow[validity] = 0
        self._optical_flow = optical_flow
        self._validity = validity

    @property
    def validity(self) -> torch.Tensor:
        if self._validity is None:
            return torch.ones_like(self.u, dtype=torch.bool)
        return self._validity

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
        of = self.__class__(self._optical_flow / (self.r.max() + epsilon))
        i = (of.a + 1) / 2 * (len(color_wheel) - 1)
        color = 255 - of.r.unsqueeze(-1) * (255 - color_wheel[i])
        return color.type(torch.uint8)


class SerializableOpticalFlow(OpticalFlow):
    SUFFIXES: Container[str]

    @classmethod
    def _validate(cls, path: pathlib.Path) -> None:
        assert path.suffix in cls.SUFFIXES

    @classmethod
    @abstractmethod
    def load(cls, path: pathlib.Path) -> Self:
        pass

    @abstractmethod
    def dump(self, path: pathlib.Path) -> None:
        pass


class FloOpticalFlow(SerializableOpticalFlow):
    SUFFIXES = {'.flo'}
    MAGIC = 202021.25

    @classmethod
    def load(cls, path: pathlib.Path) -> Self:
        cls._validate(path)
        with path.open('rb') as f:
            magic = np.fromfile(f, '<f', 1)
            assert magic == cls.MAGIC
            w = np.fromfile(f, '<i', 1).item()
            h = np.fromfile(f, '<i', 1).item()
            shape = (h, w, 2)
            data = np.fromfile(f, '<f', math.prod(shape)).reshape(shape)
        return cls(torch.tensor(data))

    def dump(self, path: pathlib.Path) -> None:
        self._validate(path)
        with path.open('wb') as f:
            np.array([self.MAGIC], '<f').tofile(f)
            np.array([self.w], '<i').tofile(f)
            np.array([self.h], '<i').tofile(f)
            data: npt.NDArray[np.float32] = self._optical_flow.numpy()
            data.astype('<f').tofile(f)


class PfmOpticalFlow(SerializableOpticalFlow):
    SUFFIXES = {'.pfm'}
    HEADER = b'PF'

    @classmethod
    def load(cls, path: pathlib.Path) -> Self:
        cls._validate(path)
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

    def dump(self, path: pathlib.Path) -> None:
        self._validate(path)
        with path.open('wb') as f:
            f.write(self.HEADER + b'\n')
            f.write(f'{self.w} {self.h}\n'.encode())
            f.write(b'-1.0\n')
            data: npt.NDArray[np.float32] = np.zeros((self.h, self.w, 3), '<f')
            data[:, :, :2] = self._optical_flow.flipud().numpy()
            data.tofile(f)


class PngOpticalFlow(SerializableOpticalFlow):
    SUFFIXES = {'.png'}

    @classmethod
    def load(cls, path: pathlib.Path) -> Self:
        cls._validate(path)
        data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        tensor = torch.tensor(data)
        validity, tensor = tensor.split_with_sizes([1, 2], dim=-1)
        validity = einops.rearrange(validity.bool(), 'h w 1 -> h w')
        tensor = tensor.float().flip(-1)
        tensor = (tensor - 2**15) / 64.
        return cls(tensor, validity)

    def dump(self, path: pathlib.Path) -> None:
        self._validate(path)
        tensor = self._optical_flow * 64 + 2**15
        tensor = tensor.flip(-1).type(torch.uint16)
        validity = einops.rearrange(
            self.validity.type(torch.uint16),
            'h w -> h w 1',
        )
        tensor = torch.cat([validity, tensor], dim=-1)
        data = tensor.numpy()
        cv2.imwrite(str(path), data)
