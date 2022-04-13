import logging
import os
import socket
from typing import Optional


_logger_initialized = False


def get_logger(log_file: Optional[str] = None, level: str = 'DEBUG'):
    global _logger_initialized
    if not _logger_initialized:
        _logger_initialized = True
        logger = logging.getLogger('Todd')
        logger.setLevel(getattr(logging, level))
        worker_pid = f"{socket.gethostname()}:{os.getpid()}"
        formatter = logging.Formatter(
            fmt=(
                f"[{worker_pid:s}][%(asctime)s.%(msecs)d]"
                f"[%(filename)s:%(lineno)d]%(levelname)s: "
                f"%(message)s"
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
