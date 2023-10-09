from todd.utils.enums import SGR


class TestSGR:

    def test_format(self) -> None:
        assert SGR.format(
            'hello',
            SGR.UNDERLINE,
            SGR.FG_GREEN,
            SGR.BG_GREEN,
        ) == '\033[4;32;42mhello\033[m'
