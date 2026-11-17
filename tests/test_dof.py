import pytest

from postfx.dof import (
    circle_of_confusion,
    bokeh_kernel,
    composite,
    autofocus
)


def test_coc_at_focus():
    assert circle_of_confusion(50.0, 2.0, 10.0, 10.0) == 0.0

def test_coc_far():
    assert circle_of_confusion(50.0, 2.0, 10.0, 20.0) == pytest.approx(12.5)

def test_coc_near():
    assert circle_of_confusion(50.0, 2.0, 10.0, 5.0) == pytest.approx(12.5)

def test_coc_wider_aperture():
    assert circle_of_confusion(50.0, 1.4, 10.0, 20.0) > circle_of_confusion(50.0, 8.0, 10.0, 20.0)

def test_bokeh_length():
    assert len(bokeh_kernel(2, 8)) == 8

def test_bokeh_weights_sum():
    assert sum(p[2] for p in bokeh_kernel(2, 8)) == pytest.approx(1.0)

def test_bokeh_symmetric():
    assert (2, 0) in [p[:2] for p in bokeh_kernel(2, 8)] and (-2, 0) in [p[:2] for p in bokeh_kernel(2, 8)]

def test_composite_sharp():
    assert composite((0.2, 0.4, 0.6), (1.0, 1.0, 1.0), 0.0, 10.0) == (0.2, 0.4, 0.6)

def test_composite_full_blur():
    assert composite((0.2, 0.4, 0.6), (1.0, 1.0, 1.0), 10.0, 10.0) == (1.0, 1.0, 1.0)

def test_composite_half():
    assert composite((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 5.0, 10.0) == pytest.approx((0.5, 0.5, 0.5))

def test_autofocus_moves():
    assert autofocus(0.0, 10.0, 0.1, 5.0) == pytest.approx(0.5)

def test_autofocus_clamps():
    assert autofocus(9.8, 10.0, 0.1, 5.0) == 10.0

def test_autofocus_zero_dt():
    assert autofocus(0.0, 10.0, 0.0, 5.0) == 0.0

def test_coc_negative_depth_uses_abs():
    assert circle_of_confusion(50.0, 2.0, 10.0, 5.0) > 0.0

def test_composite_energy_preserved():
    assert composite((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 5.0, 10.0) == (1.0, 1.0, 1.0)
