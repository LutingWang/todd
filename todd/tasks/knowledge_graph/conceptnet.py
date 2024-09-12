__all__ = [
    'ConceptNet',
]

from typing import cast

import pandas as pd
import torch


class ConceptNet:

    def __init__(self) -> None:
        df = pd.read_hdf('data/conceptnet/mini.h5')
        self._embeddings = torch.from_numpy(df.to_numpy())
        self._embedding_indices = {id_: i for i, id_ in enumerate(df.index)}

    def embedding(self, key: str) -> torch.Tensor:
        index = self._embedding_indices[key]
        return self._embeddings[index]

    def similarity(self, key1: str, key2: str) -> int:
        embedding1 = self.embedding(key1).int()
        embedding2 = self.embedding(key2).int()
        similarity = torch.einsum('i, i -> ', embedding1, embedding2)
        return cast(int, similarity.item())
