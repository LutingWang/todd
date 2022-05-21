from abc import abstractmethod
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import List, Optional
import cv2

import numpy as np

from ..utils.iters import get_iter

from .base import BaseVisual
from .builder import VISUALS


class BaseSaver(BaseVisual):
    def __init__(
        self, 
        *args, 
        root_dir: str, 
        trial_name: Optional[str] = None, 
        suffix: str = '',
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._root_dir = Path(root_dir)
        self._trial_name = datetime.now().strftime("%Y%m%dT%H%M%S%f") if trial_name is None else trial_name
        self._suffix = suffix

        visual_dir = self._root_dir / self._trial_name
        if not visual_dir.exists():
            visual_dir.mkdir(parents=True)
        self._visual_dir = visual_dir

    def _get_path(self, as_str: bool = False, **kwargs) -> PathLike:
        assert 'iter' not in kwargs
        kwargs['iter'] = get_iter()
        kwargs = '_'.join(f'{k}-{v}' for k, v in kwargs.items())
        path = self._visual_dir / f'{kwargs}_{self._suffix}'
        if as_str:
            return str(path)
        return path

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


@VISUALS.register_module()
class CV2Saver(BaseSaver):
    def __init__(self, *args, suffix: str = '', **kwargs):
        super().__init__(*args, suffix=suffix + '.png', **kwargs)

    def forward(self, images: List[np.ndarray], **kwargs):
        for sample, image in enumerate(images):
            path = self._get_path(as_str=True, sample=sample, **kwargs)
            result = cv2.imwrite(path, image)
            assert result, path
