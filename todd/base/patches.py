import builtins

from .logger import logger

try:
    import ipdb
    logger.info("`ipdb` is installed. Using it for debugging.")
    builtins.breakpoint = ipdb.set_trace
except ImportError:
    pass
