from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import VISUALS, BaseVisual

__all__ = [
    'AnnotationVisual',
]


@VISUALS.register_module()
class AnnotationVisual(BaseVisual):
    PALETTE = [  # yapf: disable
        (106,   0, 228), (119,  11,  32), (165,  42,  42), (  0,   0, 192),
        (197, 226, 255), (  0,  60, 100), (  0,   0, 142), (255,  77, 255),
        (153,  69,   1), (120, 166, 157), (  0, 182, 199), (  0, 226, 252),
        (182, 182, 255), (  0,   0, 230), (220,  20,  60), (163, 255,   0),
        (  0,  82,   0), (  3,  95, 161), (  0,  80, 100), (183, 130,  88),
    ]

    # PALETTE = [
    #     (220,  20,  60), (119,  11,  32), (  0,   0, 142), (  0,   0, 230),
    #     (106,   0, 228), (  0,  60, 100), (  0,  80, 100), (  0,   0,  70),
    #     (  0,   0, 192), (250, 170,  30), (100, 170,  30), (220, 220,   0),
    #     (175, 116, 175), (250,   0,  30), (165,  42,  42), (255,  77, 255),
    #     (  0, 226, 252), (182, 182, 255), (  0,  82,   0), (120, 166, 157),
    #     (110,  76,   0), (174,  57, 255), (199, 100,   0), ( 72,   0, 118),
    #     (255, 179, 240), (  0, 125,  92), (209,   0, 151), (188, 208, 182),
    #     (  0, 220, 176), (255,  99, 164), ( 92,   0,  73), (133, 129, 255),
    #     ( 78, 180, 255), (  0, 228,   0), (174, 255, 243), ( 45,  89, 255),
    #     (134, 134, 103), (145, 148, 174), (255, 208, 186), (197, 226, 255),
    #     (171, 134,   1), (109,  63,  54), (207, 138, 255), (151,   0,  95),
    #     (  9,  80,  61), ( 84, 105,  51), ( 74,  65, 105), (166, 196, 102),
    #     (208, 195, 210), (255, 109,  65), (  0, 143, 149), (179,   0, 194),
    #     (209,  99, 106), (  5, 121,   0), (227, 255, 205), (147, 186, 208),
    #     (153,  69,   1), (  3,  95, 161), (163, 255,   0), (119,   0, 170),
    #     (  0, 182, 199), (  0, 165, 120), (183, 130,  88), ( 95,  32,   0),
    #     (130, 114, 135), (110, 129, 133), (166,  74, 118), (219, 142, 185),
    #     ( 79, 210, 114), (178,  90,  62), ( 65,  70,  15), (127, 167, 115),
    #     ( 59, 105, 106), (142, 108,  45), (196, 172,   0), ( 95,  54,  80),
    #     (128,  76, 255), (201,  57,   1), (246,   0, 122), (191, 162, 208),
    # ]

    def get_palette(self, index: int) -> Tuple[int, int, int]:
        index %= len(self.PALETTE)
        return self.PALETTE[index]

    def forward(
        self,
        images: List[np.ndarray],
        bboxes: List[np.ndarray],
        classes: List[np.ndarray],
        class_names: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """Annotation visualizer.

        Args:
            images: :math:`(N, H, W, 3)`
        """
        results = []
        for image, bbox, class_ in zip(images, bboxes, classes):
            image = np.ascontiguousarray(image)
            bbox = bbox.astype(np.int32)
            class_ = class_.astype(np.int32)
            for (l, t, r, b), c in zip(bbox.tolist(), class_.tolist()):
                color = self.get_palette(c)
                cv2.rectangle(image, (l, t), (r, b), color, thickness=1)
                class_name = (
                    str(c) if class_names is None else class_names[c]
                )
                cv2.putText(
                    image,
                    text=class_name,
                    org=(l, t),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5,
                    color=color,
                )
            results.append(image)

        return results
