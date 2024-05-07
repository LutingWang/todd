__all__ = [
    'get_timestamp',
    'subprocess_run',
    'descendant_classes',
]

import subprocess  # nosec B404
from datetime import datetime


def get_timestamp() -> str:
    timestamp = datetime.now().astimezone().isoformat()
    timestamp = timestamp.replace(':', '-')
    timestamp = timestamp.replace('+', '-')
    timestamp = timestamp.replace('.', '_')
    return timestamp


def subprocess_run(args: str) -> str:
    return subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,  # nosec B602
        text=True,
    ).stdout


def descendant_classes(cls: type) -> list[type]:
    classes = []
    for subclass in cls.__subclasses__():
        classes.append(subclass)
        classes.extend(descendant_classes(subclass))
    return classes
