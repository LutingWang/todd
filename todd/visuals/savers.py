from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Generic, Optional, Sequence, TypeVar

import cv2
import numpy as np

from ..base import get_iter
from .base import VISUALS, BaseVisual

__all__ = [
    'BaseSaver',
    'CV2Saver',
]

T = TypeVar('T')


class BaseSaver(Generic[T], BaseVisual):

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
        self._trial_name = (
            datetime.now().strftime("%Y%m%dT%H%M%S%f")
            if trial_name is None else trial_name
        )
        self._suffix = suffix

        visual_dir = self._root_dir / self._trial_name
        if not visual_dir.exists():
            visual_dir.mkdir(parents=True)
        self._visual_dir = visual_dir

    def _get_path(self, **kwargs) -> Path:
        assert 'iter' not in kwargs
        kwargs['iter'] = get_iter()
        filename = '_'.join(f'{k}-{v}' for k, v in kwargs.items())
        path = self._visual_dir / f'{filename}_{self._suffix}'
        return path

    @abstractmethod
    def forward(self, data: Sequence[T], **kwargs) -> None:
        pass


@VISUALS.register_module()
class CV2Saver(BaseSaver[np.ndarray]):

    def __init__(self, *args, suffix: str = '', **kwargs):
        super().__init__(*args, suffix=suffix + '.png', **kwargs)

    def forward(self, data: Sequence[np.ndarray], **kwargs) -> None:
        for sample, image in enumerate(data):
            path = self._get_path(sample=sample, **kwargs)
            result = cv2.imwrite(str(path), image)
            assert result, path
