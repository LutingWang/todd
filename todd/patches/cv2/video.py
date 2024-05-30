__all__ = [
    'VideoWriter',
]

import argparse
import atexit
import os
import tempfile
from typing import Any

import cv2
import ffmpeg
import numpy as np
import numpy.typing as npt


class VideoWriter:

    def __init__(self, fps: float) -> None:
        self._fps = fps

        fd, file = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        atexit.register(self._cleanup)
        self._file = file

        self._video_writer = cv2.VideoWriter()

    def _cleanup(self) -> None:
        self._video_writer.release()
        if os.path.exists(self._file):
            os.remove(self._file)

    @property
    def opened(self) -> bool:
        return self._video_writer.isOpened()

    def write(self, frame: npt.NDArray[np.uint8]) -> None:
        if not self.opened:
            assert os.path.exists(self._file)
            h, w, _ = frame.shape
            self._video_writer.open(
                self._file,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self._fps,
                (w, h),
            )
        self._video_writer.write(frame)

    def dump(self, f: Any) -> None:
        self._video_writer.release()
        stream: ffmpeg.Stream = ffmpeg.input(self._file)
        stream = stream.output(str(f), vcodec='libx264')
        stream.run()


def images_to_video_cli() -> None:
    parser = argparse.ArgumentParser(description="Write Video")
    parser.add_argument('images', nargs='+')
    parser.add_argument('--fps', type=float, default=12.)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    writer = VideoWriter(args.fps)
    for image in args.images:
        frame = cv2.imread(image)
        writer.write(frame)
    writer.dump(args.out)
