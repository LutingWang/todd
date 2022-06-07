import getpass
import logging
import os
import socket
from typing import Optional

__all__ = [
    'get_logger',
]

_logger_initialized = False


def get_logger(log_file: Optional[str] = None, level: str = 'DEBUG'):
    global _logger_initialized
    if not _logger_initialized:
        from .._base import SGR

        _logger_initialized = True
        logger = logging.getLogger('Todd')
        logger.setLevel(getattr(logging, level))
        worker_pid = f"{getpass.getuser()}@{socket.gethostname()}:{os.getpid()}"
        formatter = logging.Formatter(
            fmt=(  # yapf: disable
                f"[{worker_pid:s}][%(asctime)s]"
                f"[%(filename)s:%(funcName)s:%(lineno)d] "
                + SGR.format(
                    "%(name)s %(levelname)s:", sgr=(SGR.BOLD, SGR.FG_BLUE),
                ) +
                f"\n%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.propagate = False
    return logging.getLogger('Todd')
