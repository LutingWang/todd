__all__ = [
    'BaseDataset',
]

from abc import abstractmethod
from typing import Any, TypeVar

from ....datasets import BaseDataset as BaseDataset_
from ....patches.py import classproperty
from ....registries import BuildSpec
from ..optical_flow import OpticalFlow
from ..registries import OFEDatasetRegistry

T = dict[str, Any]
VT = TypeVar('VT', bound=OpticalFlow)


@OFEDatasetRegistry.register_()
class BaseDataset(BaseDataset_[T, str, VT]):

    @classproperty
    def build_spec(self) -> BuildSpec:
        build_spec: BuildSpec = super().build_spec
        build_spec.pop('access_layer')  # pylint: disable=no-member
        return build_spec

    @abstractmethod
    def _next_key(self, key: str) -> str:
        pass
