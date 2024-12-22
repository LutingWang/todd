__all__ = [
    'Statistics',
    'Statistician',
    'fid',
]

from functools import partial
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed
import torch.nn.functional as F
import torchvision
from scipy import linalg
from torch import nn

import todd
from todd import Config
from todd.patches.torch import all_gather_object
from todd.registries import InitWeightsMixin
from todd.utils import set_temp

from .registries import IGModelRegistry


class InceptionRegistry(IGModelRegistry):
    pass


@InceptionRegistry.register_()
class Inception(
    todd.models.MeanStdMixin,
    todd.models.FrozenMixin,
    InitWeightsMixin,
):

    class InceptionE(torchvision.models.inception.InceptionE):

        @set_temp(F, '.avg_pool2d', F.max_pool2d)
        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            mean=(127.5, 127.5, 127.5),
            std=(127.5, 127.5, 127.5),
            **kwargs,
        )
        inception = torchvision.models.inception_v3(
            num_classes=1008,
            aux_logits=False,
            init_weights=False,
        )

        inception.Mixed_7c.__class__ = self.InceptionE
        inception.fc = nn.Identity()

        self._inception = inception

    def init_weights(self, config: todd.Config) -> bool:
        # https://github.com/mseitzer/pytorch-fid/releases/download/
        # fid_weights/pt_inception-2015-12-05-6726825d.pth
        f = config.get('pretrained', 'pretrained/pytorch-fid/pt_inception.pth')
        state_dict: dict[str, torch.Tensor] = torch.load(f)
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        self._inception.load_state_dict(state_dict)
        return super().init_weights(config)

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.normalize(image)
        image = F.interpolate(image, size=(299, 299), mode='bilinear')

        with set_temp(
            F,
            '.avg_pool2d',
            partial(F.avg_pool2d, count_include_pad=False),
        ):
            return self._inception(image)


class Statistics(NamedTuple):
    mu: npt.NDArray[np.float32]
    sigma: npt.NDArray[np.float32]


class Statistician(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._inception = InceptionRegistry.build(
            Config(type=Inception.__name__),
        )
        self._features: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor) -> None:
        features = self._inception(images)
        self._features.append(features)

    def summarize(self) -> Statistics:
        features = torch.cat(self._features)
        features = torch.cat(all_gather_object(features))
        features = features.cpu()
        mu = features.mean(0).numpy()
        sigma = np.cov(features.numpy(), rowvar=False)
        return Statistics(mu=mu, sigma=sigma)


def fid(gt: Statistics, pred: Statistics, eps: float = 1e-6) -> float:
    delta_mu = gt.mu - pred.mu

    cov = gt.sigma.dot(pred.sigma)
    cov, _ = linalg.sqrtm(cov, disp=False)

    if np.isinf(cov).any():
        todd.logger.warning("FID calculation produces singular product")
        gt_offset = np.eye(gt.sigma.shape[0]) * eps
        offset = np.eye(pred.sigma.shape[0]) * eps
        cov = (gt.sigma + gt_offset).dot(pred.sigma + offset)
        cov = linalg.sqrtm(cov, disp=False)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(cov):
        if not np.allclose(np.diagonal(cov).imag, 0, atol=1e-3):
            message = f'Imaginary component {np.max(np.abs(cov.imag))}'
            todd.logger.error(message)
            if not todd.Store.DRY_RUN:
                raise ValueError(message)
        cov = cov.real

    return (
        delta_mu.dot(delta_mu) + np.trace(gt.sigma) + np.trace(pred.sigma)
        - 2 * np.trace(cov)
    )
