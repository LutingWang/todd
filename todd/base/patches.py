import builtins

from ..utils import get_rank
from .logger import logger

try:
    import ipdb
    if get_rank() == 0:
        logger.info("`ipdb` is installed. Using it for debugging.")
    builtins.breakpoint = ipdb.set_trace
except ImportError:
    pass
