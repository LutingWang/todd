import pathlib
from collections import defaultdict

import cv2
import numpy as np

import todd.tasks.optical_flow_estimation as ofe
from todd import Config
from todd.patches.cv2 import ColorMap, VideoWriter

data_root = pathlib.Path('data/sintel')

sintel_dataset = ofe.datasets.SintelDataset(
    access_layer=Config(data_root=data_root),
    pass_='final',  # nosec B106
)
color_map = ColorMap(cv2.COLORMAP_JET)
video_writers: defaultdict[str, VideoWriter] = \
    defaultdict(lambda: VideoWriter(12.))
for data in sintel_dataset:
    of = ofe.OpticalFlow(data['of'])
    frame = np.concatenate([
        data['frame1'],
        of.to_color().numpy(),
        color_map(of.a),
        color_map(of.r),
        color_map(of.u),
        color_map(of.v),
    ])
    scene, _ = data['id_'].split('/', 1)
    video_writers[scene].write(frame)
for name, video_writer in video_writers.items():
    video_writer.dump(f'{name}.mp4')
