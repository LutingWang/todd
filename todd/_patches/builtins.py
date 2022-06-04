import builtins

import ipdb

builtins.breakpoint = ipdb.set_trace
