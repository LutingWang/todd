__all__ = [
    'ConceptNet',
    'ConceptNetNumbersbatch',
]

from dataclasses import dataclass
from typing import Any, TypedDict, cast
from typing_extensions import Self

import pandas as pd

from todd.loggers import logger

DATA_ROOT = 'data/conceptnet'


class Source(TypedDict):
    contributor: str
    process: str


class Info(TypedDict):
    dataset: str
    license: str
    sources: list[Source]
    weight: float


@dataclass(frozen=True)
class Edge:
    uri: str
    relation: str
    start: str
    end: str
    info: Info


class ConceptNet(pd.DataFrame):

    @classmethod
    def load(
        cls,
        f: Any = f'{DATA_ROOT}/conceptnet-assertions-5.7.0.csv.gz',
    ) -> Self:
        logger.info("Loading %s", f)
        df = pd.read_csv(
            f,
            sep='\t',
            compression='gzip',
            header=None,
            names=list(Edge.__dataclass_fields__),  # noqa: E501 pylint: disable=no-member
        )
        df.__class__ = cls
        return cast(Self, df)


class ConceptNetNumbersbatch(pd.DataFrame):

    @classmethod
    def load(cls, f: Any = f'{DATA_ROOT}/mini.h5') -> Self:
        logger.info("Loading %s", f)
        df = pd.read_hdf(f)
        assert isinstance(df, pd.DataFrame)
        df.__class__ = cls
        return cast(Self, df)
