import pathlib

import numpy as np

from todd_torch.datasets.access_layers import NpyAccessLayer
from todd_torch.datasets.access_layers.concat import ConcatAccessLayer


class TestConcatAccessLayer:

    def test_len(self, tmp_path: pathlib.Path) -> None:
        first: NpyAccessLayer[np.int64] = NpyAccessLayer(
            data_root=str(tmp_path),
            task_name='first',
        )
        second: NpyAccessLayer[np.int64] = NpyAccessLayer(
            data_root=str(tmp_path),
            task_name='second',
        )
        access_layer = ConcatAccessLayer(
            access_layers={
                'first': first,
                'second': second,
            },
        )
        access_layer.touch()

        assert len(access_layer) == 0

        first['one'] = np.array(1)
        second['two'] = np.array(2)
        second['three'] = np.array(3)
        assert len(access_layer) == 3
