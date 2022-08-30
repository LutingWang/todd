import os
import shutil
import subprocess
import unittest.mock as mock

import pytest

from todd.base.misc import git_commit_id, git_status, which_git


@mock.patch.object(shutil, 'which', autospec=True)
def test_which_git(mock_which: mock.MagicMock) -> None:
    git = '/usr/bin/git'
    mock_which.return_value = git
    assert which_git() == git
    assert mock_which.call_args == (('git', ), )

    mock_which.return_value = None
    with pytest.raises(AssertionError):
        which_git()


@mock.patch.object(subprocess, 'run', autospec=True)
@mock.patch.object(shutil, 'which', autospec=True)
def test_git_status(
    mock_which: mock.MagicMock,
    mock_run: mock.MagicMock,
) -> None:
    git = 'git'
    args = [git, 'status', '--porcelain']
    mock_which.return_value = git

    mock_run.return_value = subprocess.CompletedProcess(
        args,
        returncode=0,
        stdout='',
        stderr='',
    )
    assert git_status() == []
    assert mock_run.call_args == (
        (args, ),
        dict(capture_output=True, text=True),
    )

    mock_run.return_value = subprocess.CompletedProcess(
        args,
        returncode=0,
        stdout=' M README.md\n M pyproject.toml\n',
        stderr='',
    )
    assert git_status() == [' M README.md', ' M pyproject.toml']

    mock_run.return_value = subprocess.CompletedProcess(
        args,
        returncode=128,
        stdout='',
        stderr=(
            'fatal: not a git repository '
            '(or any of the parent directories): .git\n'
        ),
    )
    assert git_status() == []


@mock.patch.object(subprocess, 'run', autospec=True)
@mock.patch.object(shutil, 'which', autospec=True)
def test_git_commit_id(
    mock_which: mock.MagicMock,
    mock_run: mock.MagicMock,
) -> None:
    git = 'git'
    args = [git, 'rev-parse', '--short', 'HEAD']
    mock_which.return_value = git

    mock_run.return_value = subprocess.CompletedProcess(
        args,
        returncode=0,
        stdout='dde5562\n',
        stderr='',
    )
    assert git_commit_id() == 'dde5562'
    assert mock_run.call_args == (
        (args, ),
        dict(capture_output=True, text=True),
    )

    mock_run.return_value = subprocess.CompletedProcess(
        args,
        returncode=128,
        stdout='',
        stderr=(
            'fatal: not a git repository '
            '(or any of the parent directories): .git\n'
        ),
    )
    assert git_commit_id() == ''

    os.environ['GIT_COMMIT_ID'] = 'dde5562'
    assert git_commit_id() == 'dde5562'
