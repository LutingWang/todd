__all__ = [
    'query',
    'buckets',
    'script',
    'cluster',
    'pai',
    'odps',
]

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

    Example:

        >>> config = Config(
        ...     role_arn='role/arn',
        ...     host='host.oss.aliyuncs.com',
        ... )
        >>> print(query(config))
        role_arn=role/arn&host=host.oss.aliyuncs.com
    """
    return urlencode(config, safe='/:')


def buckets(config: Config) -> str:
    """Encode the bucket url.

    Args:
        config: the bucket configuration.

    Returns:
        The encoded bucket url.

    Example:

        >>> config = Config(
        ...     scheme='oss',
        ...     netloc='data',
        ...     path='/datasets/',
        ...     fragment='',
        ...     query=dict(
        ...         role_arn='role/arn',
        ...         host='host.oss.aliyuncs.com',
        ...     ),
        ... )
        >>> print(buckets(config))
        oss://data/datasets/?role_arn=role/arn&host=host.oss.aliyuncs.com
    """
    config.query = query(config.query)
    return SplitResult(**config).geturl()


def script(config: Config) -> str:
    """Compress the files and return the compressed file path.

    Args:
        config: the script configuration.

    Returns:
        The compressed file path.

    Example:

        >>> config = Config(
        ...     name='todd',
        ...     files=['todd', 'pyproject.toml'],  # add `requirements.txt`
        ...     filters=['__pycache__'],
        ... )
        >>> print(script(config))
        file:///tmp/todd.tar.gz
    """
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
    """Encode the cluster configuration.

    Args:
        config: the cluster configuration.

    Returns:
        The encoded cluster configuration.

    Example:

        >>> config = Config(worker=dict(
        ...     gpu=200,
        ...     cpu=1600,
        ...     memory=100000,
        ... ))
        >>> print(cluster(config))
        {\\"worker\\": {\\"gpu\\": 200, \\"cpu\\": 1600, \\"memory\\": 100000}}
    """
    result = str(config)
    return result.replace('\'', '\\"')


def pai(config: Config) -> str:
    """Encode the PAI configuration.

    Args:
        config: the PAI configuration.

    Returns:
        The encoded configuration.

    Example:

        >>> config = Config(
        ...     name='pytorch180',
        ...     oversubscription=False,
        ...     entry_file='tools/launch.py',
        ...     worker_count=4,
        ...     user_defined_parameters=['main.py'],
        ...     buckets=dict(
        ...         scheme='oss',
        ...         netloc='data',
        ...         path='/datasets/',
        ...         fragment='',
        ...         query=dict(
        ...             role_arn='role/arn',
        ...             host='host.oss.aliyuncs.com',
        ...         ),
        ...     ),
        ...     script=dict(
        ...         name='todd',
        ...         files=['todd', 'pyproject.toml'],
        ...         filters=['__pycache__'],
        ...     ),
        ...     cluster=dict(worker=dict(
        ...         gpu=200,
        ...         cpu=1600,
        ...         memory=100000,
        ...     )),
        ... )
        >>> print(pai(config))
        <BLANKLINE>
        pai -name pytorch180
            -Doversubscription=false
            -Dscript="file:///tmp/todd.tar.gz"
            -DentryFile="tools/launch.py"
            -DworkerCount=4
            -DuserDefinedParameters="main.py"
            -Dbuckets="oss://data/datasets/?role_arn=role/arn&host=host.oss.al\
iyuncs.com"
            -Dcluster="{\\"worker\\": {\\"gpu\\": 200, \\"cpu\\": 1600, \\"mem\
ory\\": 100000}}";
        <BLANKLINE>
    """
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
    """Convert the configuration to an ODPS command.

    Args:
        config: the ODPS configuration.

    Returns:
        The converted ODPS command.

    Example:

        >>> config = Config(
        ...     workbench='dev',
        ...     gpu='V100M32',
        ...     pai=dict(
        ...         name='pytorch180',
        ...         oversubscription=False,
        ...         entry_file='tools/launch.py',
        ...         worker_count=4,
        ...         user_defined_parameters=['main.py'],
        ...         buckets=dict(
        ...             scheme='oss',
        ...             netloc='data',
        ...             path='/datasets/',
        ...             fragment='',
        ...             query=dict(
        ...                 role_arn='role/arn',
        ...                 host='host.oss.aliyuncs.com',
        ...             ),
        ...         ),
        ...         script=dict(
        ...             name='todd',
        ...             files=['todd', 'pyproject.toml'],
        ...             filters=['__pycache__'],
        ...         ),
        ...         cluster=dict(worker=dict(
        ...             gpu=200,
        ...             cpu=1600,
        ...             memory=100000,
        ...         )),
        ...     ),
        ... )
        >>> print(odps(config))
        use dev;
        set odps.algo.hybrid.deploy.info=LABEL:V100M32:OPER_EQUAL;
        pai -name pytorch180
            -Doversubscription=false
            -Dscript="file:///tmp/todd.tar.gz"
            -DentryFile="tools/launch.py"
            -DworkerCount=4
            -DuserDefinedParameters="main.py"
            -Dbuckets="oss://data/datasets/?role_arn=role/arn&host=host.oss.aliyuncs.com"
            -Dcluster="{\\"worker\\": {\\"gpu\\": 200, \\"cpu\\": 1600, \\"mem\
ory\\": 100000}}";
        <BLANKLINE>
    """
    result = f'use {config.workbench};'
    if 'gpu' in config:
        result += (
            '\nset odps.algo.hybrid.deploy.info='
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
