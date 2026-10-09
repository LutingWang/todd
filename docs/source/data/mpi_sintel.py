from pathlib import Path

import todd_tasks.optical_flow_estimation as ofe
from todd import Config
from todd.configs import PyConfig

dataset = ofe.datasets.SintelDataset(
    access_layer=Config(directory=Path('data', 'sintel')),
    pass_='final',  # nosec B106
)
PyConfig.load(
    Path(__file__).parent / 'optical_flow.py',
).visualize(dataset)
