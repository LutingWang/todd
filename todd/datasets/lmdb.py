# import io
# from functools import cached_property
# from typing import Any, Literal

# import lmdb
# import torch

# from ..base import DatasetRegistry
# from .base import BaseDataset

# TODO: update

# @DatasetRegistry.register()
# class LmdbDataset(BaseDataset[bytes]):

#     def __init__(
#         self,
#         *args,
#         filepath: str,
#         db: str | None = None,
#         decoder: Literal['None', 'pytorch'] | None = 'pytorch',
#         **kwargs,
#     ):
#         self._env: lmdb.Environment = lmdb.open(
#             filepath, readonly=True, max_dbs=1
#         )
#         self._db: lmdb._Database | None = (  # yapf: disable
#             None if db is None else self._env.open_db(db.encode())
#         )
#         self._decoder = decoder
#         super().__init__(*args, **kwargs)

#     @classmethod
#     def load_from(cls, source: BaseDataset, *args, **kwargs):
#         pass

#     def _map_indices(self) -> list[bytes]:
#         with self.begin() as txn, txn.cursor() as cur:
#             return list(cur.iternext(keys=True, values=False))

#     @cached_property
#     def _len(self) -> int:
#         with self.begin() as txn:
#             return txn.stat()['entries']

#     def _getitem(self, index: bytes) -> Any:
#         if not isinstance(index, bytes):
#             index = str(index).encode()
#         with self.begin() as txn:
#             buffer = txn.get(index)
#         if self._decoder is None or self._decoder == 'None':
#             return buffer
#         if self._decoder == 'pytorch':
#             buffer = io.BytesIO(buffer)
#             return torch.load(buffer, map_location='cpu')
#         raise Exception

#     def begin(self) -> lmdb.Transaction:
#         return self._env.begin(self._db)
