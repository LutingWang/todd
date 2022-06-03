from typing import Union

from ..utils import build_metas
from .base import BaseDistiller

DISTILLERS, build_distiller = build_metas('distillers', BaseDistiller)
DistillerConfig = Union[BaseDistiller, dict]
