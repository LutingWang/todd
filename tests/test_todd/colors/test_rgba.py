from typing import Any, cast

import pytest

from todd.colors.rgba import RGB, RGBA


class TestRGB:

    def test_init(self) -> None:
        rgb = RGB(red=1., green=0., blue=0.)

        assert rgb.to_tuple() == (1., 0., 0.)

        with pytest.raises(AssertionError):
            RGB(red=255, green=0, blue=0)

    def test_from(self) -> None:
        rgb = RGB.from_('#8040FF')

        assert rgb.to_tuple() == pytest.approx((
            128 / 255,
            64 / 255,
            1.,
        ))

    def test_from_tuple(self) -> None:
        rgb = RGB.from_tuple((255, 64, 0), normalized=False)
        bgr = RGB.from_tuple(
            (0, 64, 255),
            normalized=False,
            order='bgr',
        )

        assert rgb.to_tuple() == pytest.approx((1., 64 / 255, 0.))
        assert bgr.to_tuple() == pytest.approx(rgb.to_tuple())

        with pytest.raises(ValueError):
            RGB.from_tuple((1., 0., 0.), order=cast(Any, 'invalid'))

    def test_to_tuple(self) -> None:
        rgb = RGB(red=.5, green=.25, blue=1.)

        assert rgb.to_tuple() == (.5, .25, 1.)
        assert rgb.to_tuple(normalized=False) == (127, 63, 255)
        assert rgb.to_tuple(
            normalized=False,
            order='bgr',
        ) == (255, 63, 127)

        with pytest.raises(ValueError):
            rgb.to_tuple(order=cast(Any, 'invalid'))


class TestRGBA:

    def test_to_tuple(self) -> None:
        rgba = RGBA(red=.5, green=.25, blue=1., alpha=.5)

        assert rgba.to_tuple() == (.5, .25, 1., .5)
        assert rgba.to_tuple(normalized=False) == (127, 63, 255, 127)
        assert rgba.to_tuple(
            normalized=False,
            order='bgr',
        ) == (255, 63, 127, 127)
