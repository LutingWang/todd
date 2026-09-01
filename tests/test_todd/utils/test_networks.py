from unittest import mock

import requests

from todd.utils.networks import get_bytes


def test_get_bytes() -> None:
    response = mock.Mock()
    response.content = b'hello'
    with mock.patch.object(requests, 'get', return_value=response) as get:
        assert get_bytes('https://example.com').read() == b'hello'
        get.assert_called_once_with('https://example.com', timeout=5)
        response.raise_for_status.assert_called_once_with()
