import argparse
import os
import re
import subprocess
import tarfile
from urllib.parse import SplitResult, urlencode

from ..base import Config, DictAction, logger


def query(config: Config) -> str:
    """Encode the bucket url query.

    Args:
        config: query configuration.

    Returns:
        Encoded query.
    """
    return urlencode(config, safe='/:')


def buckets(config: Config) -> str:
    config.query = query(config.query)
    return SplitResult(**config).geturl()


def script(config: Config) -> str:
    result = f'/tmp/{config.name}.tar.gz'

    def filter_(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if any(re.match(f, info.name) for f in config.filters):
            return None
        return info

    with tarfile.open(result, 'w:gz') as file:
        for f in config.files:
            file.add(f, filter=filter_)
    return 'file://' + result


def cluster(config: Config) -> str:
    result = str(config)
    return result.replace('\'', '\\"')


def pai(config: Config) -> str:
    return f'''
pai -name {config.name}
    -Doversubscription={str(config.oversubscription).lower()}
    -Dscript="{script(config.script)}"
    -DentryFile="{config.entry_file}"
    -DworkerCount={config.worker_count}
    -DuserDefinedParameters="{' '.join(config.user_defined_parameters)}"
    -Dbuckets="{buckets(config.buckets)}"
    -Dcluster="{cluster(config.cluster)}";
'''


def odps(config: Config) -> str:
    result = f'use {config.workbench};'
    if 'gpu' in config:
        result += (
            'set odps.algo.hybrid.deploy.info='
            f'LABEL:{config.gpu}:OPER_EQUAL;'
        )
    result += pai(config.pai)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ODPS run')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--override', action=DictAction)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    if os.path.exists('.pre-commit-config.yaml'):
        subprocess.check_call(['pre-commit', 'run', 'debug-statements', '-a'])
    else:
        logger.info("pre-commit config does not exist")

    if git_status := subprocess.check_output(['git', 'status', '--porcelain']):
        if args.force:
            logger.warning(git_status)
        else:
            raise RuntimeError(git_status)

    config = Config.load('pyproject.toml', override=args.override)
    command = odps(config.odps)
    logger.info(f"Running\n{command}")

    try:
        subprocess.run(['odpscmd', '-e', command])
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
