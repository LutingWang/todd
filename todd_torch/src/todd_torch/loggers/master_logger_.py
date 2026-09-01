__all__ = [
    'master_logger',
]

import logging

from ..patches.torch import get_rank

master_logger = logging.getLogger('todd.master')
if get_rank() > 0:
    master_logger.disabled = True
