from typing import Any

from mmcv.runner import BaseModule

from ..datasets import AccessLayerConfig, build_access_layer
from .builder import DISTILLERS
from .teacher import SingleTeacherDistiller


class PreprocessedTeacher(BaseModule):

    def __init__(self, *args, access_layer: AccessLayerConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self._access_layer = build_access_layer(access_layer)

    def forward(self, key: Any) -> Any:
        return self._access_layer[key]


@DISTILLERS.register_module()
class PreprocessedDistiller(SingleTeacherDistiller):
    teacher: PreprocessedTeacher

    def __init__(self, *args, teacher_cfg: AccessLayerConfig, **kwargs):
        teacher = PreprocessedTeacher(access_layer=teacher_cfg)
        super().__init__(  # type: ignore[misc]
            *args,
            teacher=teacher,
            teacher_hooks=None,
            teacher_trackings=None,
            teacher_online=False,
            **kwargs,
        )
