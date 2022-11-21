#!python
import argparse
import re
import tarfile
from typing import Any, Dict, Optional
from urllib.parse import SplitResult, urlencode

import toml


def query(config: Dict[str, str]) -> str:
    return urlencode(config, safe='/:')


def buckets(config: Dict[str, Any]) -> str:
    config['query'] = query(config['query'])
    return SplitResult(**config).geturl()


def script(config: Dict[str, Any]) -> str:
    result = f'/tmp/{config["name"]}.tar.gz'

    def filter_(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        if any(re.match(f, info.name) for f in config['filters']):
            return None
        return info

    with tarfile.open(result, 'w:gz') as file:
        for f in config['files']:
            file.add(f, filter=filter_)
    return 'file://' + result


def pai(config: Dict[str, Any]) -> str:
    parameters = [
        '--odps',
        rf'GIT_HEAD::{args.git_head}',
    ]
    parameters.extend(args.debug)
    parameters = config['user_defined_parameters'] + parameters

    return rf'''
pai -name {config["name"]}
    -Doversubscription={str(config["oversubscription"]).lower()}
    -Dscript="{script(config["script"])}"
    -DentryFile="{config["entry_file"]}"
    -DworkerCount={config["worker_count"]}
    -DuserDefinedParameters="{" ".join(parameters)}"
    -Dbuckets="{buckets(config["buckets"])}";
'''


def odps(config: Dict[str, Any]) -> str:
    return f'use {config["workbench"]};' + pai(config['pai'])


parser = argparse.ArgumentParser('odpsrun')
parser.add_argument('git_head', help='git head id')
parser.add_argument('debug', nargs='*', help='debug options')
args = parser.parse_args()

config = toml.load('pyproject.toml')['odps']
print(odps(config))
