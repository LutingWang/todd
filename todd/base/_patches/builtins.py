import builtins

from ..logger import get_logger

_logger = get_logger()

try:
    import ipdb

    _logger.info("`ipdb` is installed. Using it for debugging.")
    builtins.breakpoint = ipdb.set_trace
except ImportError:
    pass
