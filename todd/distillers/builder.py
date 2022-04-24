from ..utils import build_metas

from .base import BaseDistiller


DISTILLERS, DistillerConfig, build_distiller = build_metas('distillers', BaseDistiller)
