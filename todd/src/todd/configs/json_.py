__all__ = [
    'JsonConfig',
]

import json
from typing import Any

from .config import Config
from .registries import ConfigRegistry
from .serialize import SerializeMixin


@ConfigRegistry.register_('json')
class JsonConfig(SerializeMixin, Config):  # type: ignore[misc]

    @classmethod
    def _loads(cls, __s: str, **kwargs) -> dict[str, Any]:
        return json.loads(__s) | kwargs

    def dumps(self) -> str:
        return json.dumps(self, indent=4)
