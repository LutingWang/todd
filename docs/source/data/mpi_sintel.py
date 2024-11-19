import pathlib

import todd.tasks.optical_flow_estimation as ofe
from todd import Config
from todd.configs import PyConfig

dataset = ofe.datasets.SintelDataset(
    access_layer=Config(data_root='data/sintel'),
    pass_='final',  # nosec B106
)
PyConfig.load(
    pathlib.Path(__file__).parent / 'optical_flow.py',
).visualize(dataset)
