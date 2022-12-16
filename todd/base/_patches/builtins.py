import builtins

from ..loggers import get_logger

try:
    import ipdb

    get_logger().info("`ipdb` is installed. Using it for debugging.")
    builtins.breakpoint = ipdb.set_trace
except ImportError:
    pass
