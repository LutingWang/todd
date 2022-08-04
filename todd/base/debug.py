__all__ = [
    'DebugEnum',
]

import enum
import os


class DebugEnum(enum.Enum):

    @classmethod
    def is_active(cls) -> bool:
        return any(e.is_on for e in cls)

    @property
    def is_on(self) -> bool:
        return any(map(os.getenv, ['DEBUG', self.name]))

    def turn_on(self) -> None:
        os.environ[self.name] = '1'

    def turn_off(self) -> None:
        os.environ[self.name] = ''
