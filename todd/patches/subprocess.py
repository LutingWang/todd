__all__ = [
    'run',
]

import subprocess  # nosec B404


def run(args: str) -> str:
    return subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,  # nosec B602
        text=True,
    ).stdout
