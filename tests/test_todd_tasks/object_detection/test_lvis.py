import subprocess  # nosec B404
import sys


def test_lvis_does_not_override_todd_logging() -> None:
    code = '''
import logging

import todd

root_handlers = tuple(logging.getLogger().handlers)
todd_handlers = tuple(todd.logger.handlers)
root_formats = tuple(handler.formatter._fmt for handler in root_handlers)
todd_formats = tuple(handler.formatter._fmt for handler in todd_handlers)

import lvis

assert tuple(logging.getLogger().handlers) == root_handlers
assert tuple(todd.logger.handlers) == todd_handlers
assert tuple(
    handler.formatter._fmt for handler in root_handlers
) == root_formats
assert tuple(
    handler.formatter._fmt for handler in todd_handlers
) == todd_formats
'''
    subprocess.run([sys.executable, '-c', code], check=True)  # nosec B603
