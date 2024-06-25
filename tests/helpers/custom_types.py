from typing import TYPE_CHECKING, Any, Never

from torch import nn

if TYPE_CHECKING:
    CustomObject = Any
else:

    class CustomObject:

        def __init__(self, **kwargs) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)


class CustomModule(nn.Module):

    def __init__(self, **kwargs: nn.Module) -> None:
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, *args, **kwargs) -> Never:
        raise RuntimeError(f"{self.__class__.__name__} forward is called")
