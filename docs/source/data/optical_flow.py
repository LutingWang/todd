__all__ = [
    'visualize',
]

from typing import Any, Iterable

T = dict[str, Any]


def visualize(dataset: Iterable[T]) -> None:
    from collections import defaultdict

    import cv2
    import numpy as np
    import tqdm

    import todd.tasks.optical_flow_estimation as ofe
    from todd.patches.cv2 import ColorMap, VideoWriter

    color_map = ColorMap(cv2.COLORMAP_JET)
    video_writers: defaultdict[str, VideoWriter] = \
        defaultdict(lambda: VideoWriter(12.))
    for data in tqdm.tqdm(dataset):
        id_: str = data['id_']
        of = ofe.OpticalFlow(data['of'])
        frame = np.concatenate([
            data['frame1'],
            of.to_color().numpy(),
            color_map(of.a),
            color_map(of.r),
            color_map(of.u),
            color_map(of.v),
        ])
        scene, _ = id_.split('/', 1)
        video_writers[scene].write(frame)
    for name, video_writer in video_writers.items():
        video_writer.dump(f'{name}.mp4')
