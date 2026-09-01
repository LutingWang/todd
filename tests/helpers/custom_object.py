from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    CustomObject = Any
else:

    class CustomObject:

        def __init__(self, **kwargs) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)
