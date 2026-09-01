from typing import Never

from torch import nn


class CustomModule(nn.Module):

    def __init__(self, **kwargs: nn.Module) -> None:
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, *args, **kwargs) -> Never:
        raise RuntimeError(f"{self.__class__.__name__} forward is called")
