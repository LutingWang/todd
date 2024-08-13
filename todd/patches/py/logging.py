__all__ = [
    'Formatter',
]

import logging

# prevent lvis from overriding the logging config
import lvis  # noqa: F401 pylint: disable=unused-import

logging.basicConfig(force=True)


class Formatter(logging.Formatter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            "[%(asctime)s %(process)d:%(thread)d]"
            "[%(filename)s:%(lineno)d %(name)s %(funcName)s]"
            " %(levelname)s: %(message)s",
            *args,
            **kwargs,
        )
