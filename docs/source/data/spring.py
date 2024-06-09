import pathlib
from typing import Any

import todd.tasks.optical_flow_estimation as ofe
from todd import Config
from todd.configs import PyConfig

T = dict[str, Any]


def downsample(data: T) -> T:
    data['of'] = data['of'][::2, ::2]
    return data


dataset = ofe.datasets.SpringDataset(
    access_layer=Config(
        data_root='data/spring_sample',
        task_name='train',
        modality='flow_FW_left',
    ),
    frame_access_layer=Config(modality='frame_left'),
)
PyConfig.load(
    pathlib.Path(__file__).parent / 'optical_flow.py',
).visualize(map(downsample, dataset))
