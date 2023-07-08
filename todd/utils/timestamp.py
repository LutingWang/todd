__all__ = ['get_timestamp']

from datetime import datetime


def get_timestamp() -> str:
    timestamp = datetime.now().astimezone().isoformat()
    timestamp = timestamp.replace(':', '-')
    timestamp = timestamp.replace('+', '-')
    timestamp = timestamp.replace('.', '_')
    return timestamp
