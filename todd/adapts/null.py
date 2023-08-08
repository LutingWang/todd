from .base import AdaptRegistry, BaseAdapt


@AdaptRegistry.register()
class Null(BaseAdapt):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        return x
