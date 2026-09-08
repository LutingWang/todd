__all__ = [
    'Visual',
    'TAPVidDAVISVisual',
]

from typing import Any, Iterable, cast

import cv2
import einops
import numpy as np
import torch
import torchvision

from todd.colors import RGB, Color
from todd_tasks import optical_flow_estimation as ofe
from todd_torch.patches.cv2 import ColorMap, VideoWriter
from todd_torch.visuals import CV2Visual

from .datasets.tap_vid_davis import T as TAPVidDAVISDataType  # noqa: N811
from .points import Points


class Visual:

    def __init__(
        self,
        *args,
        video: torch.Tensor,
        target_points: Points,
        occluded: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        t_, h, w, c = video.shape
        visuals = [CV2Visual(width=w, height=h, channels=c) for _ in range(t_)]
        for frame, visual in zip(video, visuals):
            visual.image(frame.numpy())
        self._visuals = visuals

        self._target_points = target_points.denormalize()  # p * t
        self._occluded = occluded  # p * t

    def colorize(self, color_map: int = cv2.COLORMAP_JET) -> list[RGB]:
        tensor = self._target_points.to_tensor()
        tensor = tensor[:, 0]
        tensor = tensor - tensor.median(0).values
        tensor = einops.rearrange(tensor, 'n c -> 1 n c')
        of = ofe.OpticalFlow(optical_flow=tensor)
        colors = ColorMap(color_map)(of.a)
        colors = einops.rearrange(colors, '1 n c -> n c')
        return [
            RGB.from_tuple(color, normalized=False, order='bgr')
            for color in colors.tolist()
        ]

    def scatter(self, colors: Iterable[Color], size: int) -> None:
        colors = list(colors)
        sizes = [size] * len(colors)
        for t, visual in enumerate(self._visuals):
            points = cast(Points, self._target_points[:, t])
            types = [
                '*' if occluded else '.'
                for occluded in self._occluded[:, t].tolist()
            ]
            visual.scatter(
                points.to_tensor().int().tolist(),
                sizes,
                colors,
                types,
            )

    def trajectory(
        self,
        colors: Iterable[Color],
        thickness: int,
    ) -> None:
        colors = list(colors)
        for t, visual in enumerate(self._visuals, 1):
            trajectories = cast(Points, self._target_points[:, :t])
            for p in range(trajectories.shape[0]):
                occluded = self._occluded[p, :t]
                if occluded.all():
                    continue
                trajectory = cast(Points, trajectories[p, ~occluded])
                if trajectory.size < 2:
                    continue
                visual.trajectory(
                    trajectory.to_tensor().int().tolist(),
                    colors[p],
                    thickness,
                )

    def save_image(self, path: Any, *args, **kwargs) -> None:
        images = [visual.to_numpy() for visual in self._visuals]
        image = np.stack(images)
        image = einops.rearrange(image, 't h w c -> t c h w')
        tensor = torch.from_numpy(image)
        tensor = torchvision.utils.make_grid(tensor, *args, **kwargs)
        tensor = einops.rearrange(tensor, 'c h w -> h w c')
        image = tensor.numpy()
        cv2.imwrite(str(path), image)

    def save_video(self, path: Any, *args, **kwargs) -> None:
        video_writer = VideoWriter(*args, **kwargs)
        for visual in self._visuals:
            frame = visual.to_numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        video_writer.dump(path)


class TAPVidDAVISVisual(Visual):

    def __init__(
        self,
        *args,
        t: TAPVidDAVISDataType,
        **kwargs,
    ) -> None:
        video = t['video']
        _, h, w, _ = video.shape

        target_points = Points(
            t['target_points'],
            normalized=True,
            divisor=(w, h),
        )

        occluded = t['occluded']

        super().__init__(
            *args,
            video=video,
            target_points=target_points,
            occluded=occluded,
            **kwargs,
        )
