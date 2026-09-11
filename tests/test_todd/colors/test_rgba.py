from typing import Any, cast

import pytest

from todd.colors.rgba import RGB, RGBA


class TestRGB:

    def test_init(self) -> None:
        rgb = RGB(red=1., green=0., blue=0.)

        assert rgb.to_tuple() == (1., 0., 0.)

        with pytest.raises(AssertionError):
            RGB(red=255, green=0, blue=0)

    def test_from_hex(self) -> None:
        rgb = RGB.from_hex('#8040FF')

        assert rgb.to_tuple() == pytest.approx((
            128 / 255,
            64 / 255,
            1.,
        ))

    def test_from(self) -> None:
        rgb = RGB(red=.5, green=.25, blue=1)
        rgba = RGBA(red=.5, green=.25, blue=1, alpha=.5)
        rgb_ = RGB.from_(rgba)

        assert RGB.from_(rgb) is rgb
        assert rgb_.__class__ is RGB
        assert rgb_.to_tuple() == rgba.to_tuple()[:3]

    def test_to(self) -> None:
        rgb = RGB(red=.5, green=.25, blue=1)
        rgba = rgb.to(RGBA)

        assert rgb.to(RGB) is rgb
        assert rgba.__class__ is RGBA
        assert rgba.to_tuple() == (*rgb.to_tuple(), 1)

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

    def test_to_css(self) -> None:
        rgb = RGB(red=.5, green=.25, blue=1)

        assert rgb.to_css() == 'rgb(127,63,255)'


class TestRGBA:

    def test_from(self) -> None:
        rgb = RGB(red=.5, green=.25, blue=1)
        rgba = RGBA.from_(rgb)

        assert RGBA.from_(rgba) is rgba
        assert rgba.to_tuple() == (*rgb.to_tuple(), 1)

    def test_to(self) -> None:
        rgba = RGBA(red=.5, green=.25, blue=1, alpha=.5)
        rgb = rgba.to(RGB)

        assert rgba.to(RGBA) is rgba
        assert rgb.__class__ is RGB
        assert rgb.to_tuple() == rgba.to_tuple()[:3]

    def test_to_tuple(self) -> None:
        rgba = RGBA(red=.5, green=.25, blue=1., alpha=.5)

        assert rgba.to_tuple() == (.5, .25, 1., .5)
        assert rgba.to_tuple(normalized=False) == (127, 63, 255, 127)
        assert rgba.to_tuple(
            normalized=False,
            order='bgr',
        ) == (255, 63, 127, 127)

    def test_to_css(self) -> None:
        rgba = RGBA(red=.5, green=.25, blue=1, alpha=.5)

        assert rgba.to_css() == 'rgba(127,63,255,0.5)'
