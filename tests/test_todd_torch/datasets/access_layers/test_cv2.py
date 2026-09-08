import pathlib

import numpy as np

from todd_torch.datasets.access_layers.cv2 import CV2AccessLayer


class TestCV2AccessLayer:

    def test_setitem(self, tmp_path: pathlib.Path) -> None:
        access_layer = CV2AccessLayer(
            data_root=str(tmp_path),
            suffix='png',
        )
        image = np.array([[[255, 0, 0]]], dtype=np.uint8)

        access_layer['image'] = image

        assert np.array_equal(access_layer['image'], image)
