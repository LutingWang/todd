__all__ = [
    'PreprocessedDistiller',
]

from typing import Any, Union

from mmcv.runner import BaseModule

from ..datasets import ACCESS_LAYERS, BaseAccessLayer
from .base import DISTILLERS
from .teacher import SingleTeacherDistiller


class PreprocessedTeacher(BaseModule):

    def __init__(self, *args, access_layer: BaseAccessLayer, **kwargs):
        super().__init__(*args, **kwargs)
        self._access_layer = access_layer

    def forward(self, key: Any) -> Any:
        return self._access_layer[key]


@DISTILLERS.register_module()
class PreprocessedDistiller(SingleTeacherDistiller):
    teacher: PreprocessedTeacher

    def __init__(
        self,
        *args,
        access_layer: Union[BaseAccessLayer, dict],
        **kwargs,
    ):
        access_layer = ACCESS_LAYERS.build(access_layer)
        teacher = PreprocessedTeacher(access_layer=access_layer)
        super().__init__(  # type: ignore[misc]
            *args,
            teacher=teacher,
            teacher_hooks=None,
            teacher_trackings=None,
            teacher_online=False,
            **kwargs,
        )
