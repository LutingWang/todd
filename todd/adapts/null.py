from .base import BaseAdapt
from .builder import ADAPTS


@ADAPTS.register_module()
class Null(BaseAdapt):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
