from todd.utils.control_sequences import SGR, apply_sgr


class TestSGR:

    def test_apply_sgr(self) -> None:
        assert apply_sgr(
            ' hello ',
            SGR.SINGLY_UNDERLINED,
            SGR.DISPLAY_GREEN,
            SGR.BACKGROUND_GREEN,
        ) == '\033[4;32;42m hello \033[m'
