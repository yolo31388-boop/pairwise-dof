"""Depth-of-field (DOF) post-processing primitives.

The functions in this module intentionally stay small and dependency free so
they can be unit-tested in isolation.  They cover the parts of a DOF pass that
are easy to get subtly wrong:

* per-depth circle-of-confusion (CoC) evaluation,
* polygonal (aperture-shaped) bokeh sampling,
* separating near-field and far-field blur before compositing,
* energy-preserving composition of a sharp and a blurred buffer,
* a depth proxy for transparent particles that do not write opaque depth,
* damped / hysteretic autofocus.

All distances and CoC radii are expressed in the same unit (pixels for CoC,
world/scene units for depth).  ``focal_length`` is in millimetres and
``f_stop`` is an f-number (dimensionless); the ratio ``focal_length / f_stop``
is the aperture diameter used by the CoC model.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

__all__ = [
    "circle_of_confusion",
    "signed_coc",
    "bokeh_kernel",
    "composite",
    "split_near_far",
    "composite_near_far",
    "depth_proxy",
    "half_resolution_coc",
    "autofocus",
]

Color = Sequence[float]
KernelSample = tuple[float, float, float]

_EPS = 1e-12
_INFINITY = float("inf")


# ---------------------------------------------------------------------------
# CoC
# ---------------------------------------------------------------------------

def _signed_coc(
    focal_length: float,
    f_stop: float,
    focus_distance: float,
    depth: float,
) -> float:
    """Return the CoC with a sign: negative = near field, positive = far field.

    The aperture diameter is ``focal_length / f_stop``.  Near and far objects
    use their own side of the focal plane in the denominator.  This keeps the
    CoC bounded on both sides and, more importantly, gives the two sides
    opposite signs so the blur can be composited independently: a foreground
    object must never be smeared backwards onto the background.
    """
    focal_length = abs(float(focal_length))
    f_stop = float(f_stop)
    if f_stop <= 0.0:
        raise ValueError("f_stop must be greater than zero")

    focus_distance = abs(float(focus_distance))
    depth = abs(float(depth))

    if focus_distance <= _EPS and depth <= _EPS:
        return 0.0
    if abs(depth - focus_distance) <= _EPS:
        return 0.0

    aperture = focal_length / f_stop

    # Defocus is expressed relative to the larger of the two distances.  This
    # keeps near and far magnitudes symmetric around the focal plane and
    # avoids the ``inf / inf`` NaN when depth or focus distance is at the far
    # clip plane.  The limit of ``|depth - focus| / max(depth, focus)`` as
    # either distance grows without bound is 1, i.e. the full aperture.
    if math.isinf(depth) or math.isinf(focus_distance):
        return math.copysign(aperture, depth - focus_distance)

    scale = abs(depth - focus_distance) / max(depth, focus_distance)
    if scale <= _EPS:
        return 0.0

    # Negative in front of the focal plane, positive behind it.
    return math.copysign(aperture * scale, depth - focus_distance)


def circle_of_confusion(
    focal_length: float,
    f_stop: float,
    focus_distance: float,
    depth: float,
) -> float:
    """Return the unsigned circle-of-confusion radius for a scene depth.

    Parameters are ``(focal_length, f_stop, focus_distance, depth)``.  The
    result is zero at the focal plane and grows with defocus; a wider aperture
    (smaller f-number) produces a larger CoC.  Negative scene depths are
    treated as distances, matching depth-buffer conventions where the sign may
    be flipped by the projection.
    """
    return abs(_signed_coc(focal_length, f_stop, focus_distance, depth))


def signed_coc(
    focal_length: float,
    f_stop: float,
    focus_distance: float,
    depth: float,
) -> float:
    """Return a signed CoC: negative in front of focus, positive behind it."""
    return _signed_coc(focal_length, f_stop, focus_distance, depth)


# ---------------------------------------------------------------------------
# Bokeh kernel
# ---------------------------------------------------------------------------

def _snap(value: float, tolerance: float = 1e-9) -> float:
    """Collapse floating point noise so axis-aligned samples are exact.

    Kernels are usually built in pixel space where the interesting taps are at
    integer offsets.  Trigonometry such as ``2 * cos(pi)`` evaluates to
    ``-1.9999999999999998``; without snapping, a kernel lookup at ``(-2, 0)``
    would miss that tap and leak background colour into the blur.
    """
    nearest = round(value)
    if abs(value - nearest) <= tolerance:
        return float(nearest)
    return 0.0 if abs(value) <= tolerance else value


def bokeh_kernel(
    radius: float,
    samples: int,
    blades: int = 6,
    rotation: float = 0.0,
) -> list[KernelSample]:
    """Build a normalized polygonal bokeh kernel.

    A real lens aperture is a polygon with ``blades`` sides, not a perfect
    disk.  Sampling only a fixed disk (or a fixed 4-tap cross) makes bright
    highlights turn into blobs/squares and loses the aperture shape.  This
    kernel places ``samples`` taps around the perimeter of a regular
    ``blades``-gon of the given circumradius.

    Args:
        radius: Kernel radius in pixels.  The first tap lies at ``(radius, 0)``.
        samples: Number of taps returned.
        blades: Number of aperture blades.  Six is the common game default; a
            higher count approaches a circular aperture.  Because the taps
            follow a *polygon perimeter* rather than a circle, their radii
            vary between the apothem and the circumradius, which is what gives
            bright highlights their characteristic polygonal shape.
        rotation: Optional rotation of the polygon in radians.

    Returns:
        A list of ``(dx, dy, weight)`` tuples.  The weights sum to 1, which is
        what makes the subsequent blur energy preserving for HDR highlights.
    """
    n = int(samples)
    if n <= 0:
        raise ValueError("samples must be greater than zero")

    radius = abs(float(radius))
    if radius <= _EPS:
        return [(0.0, 0.0, 1.0)] * n

    sides = int(blades)
    if sides < 3:
        # A polygon needs at least three sides.  For degenerate requests fall
        # back to a triangle rather than silently producing a disk.
        sides = 3

    step = 2.0 * math.pi / sides
    apothem = math.cos(math.pi / sides)
    angle_step = 2.0 * math.pi / n
    weight = 1.0 / n

    kernel: list[KernelSample] = []
    for index in range(n):
        theta = rotation + index * angle_step

        # Radial distance from the center to the polygon edge along ``theta``.
        # Vertices sit at angles multiple of ``step`` and have radius ==
        # radius; edge midpoints sit at the apothem.
        local = (theta % step) - step * 0.5
        distance = radius * apothem / math.cos(local)

        # Positive zero snapping is essential here: cos(pi) is not exactly -1
        # for all angles and tests (and hardware) expect (-2, 0), not
        # (-2, 2.4e-16).
        x = _snap(distance * math.cos(theta))
        y = _snap(distance * math.sin(theta))
        kernel.append((x, y, weight))

    # Correct any floating point drift so HDR energy is preserved exactly for
    # the common integer-sized kernels.
    total = sum(sample[2] for sample in kernel)
    if total > 0.0:
        kernel = [(x, y, w / total) for (x, y, w) in kernel]
    return kernel


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def _color(color: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(channel) for channel in color)


def composite(
    sharp_color: Color,
    blurred_color: Color,
    coc: float,
    max_coc: float,
) -> tuple[float, ...]:
    """Blend a sharp and a blurred colour according to CoC coverage.

    The blend is a convex combination, so a constant HDR value keeps its
    energy: ``composite(c, c, ...) == c``.  Coverages outside ``[0, 1]`` are
    clamped, and the endpoints return the input colour exactly (no floating
    point drift for fully sharp / fully blurred pixels).
    """
    sharp = _color(sharp_color)
    blurred = _color(blurred_color)
    if len(sharp) != len(blurred):
        raise ValueError("sharp_color and blurred_color must have the same length")

    max_coc = abs(float(max_coc))
    if max_coc <= _EPS:
        return sharp

    coverage = abs(float(coc)) / max_coc
    coverage = min(1.0, max(0.0, coverage))

    if coverage <= 0.0:
        return sharp
    if coverage >= 1.0:
        return blurred

    return tuple(
        sharp_channel + (blurred_channel - sharp_channel) * coverage
        for sharp_channel, blurred_channel in zip(sharp, blurred)
    )


def split_near_far(coc: float, max_coc: float) -> tuple[float, float]:
    """Split a signed CoC into independent ``(near, far)`` coverages.

    Near-field and far-field blur must be accumulated in separate buffers.
    Composing a single unsigned CoC lets foreground pixels bleed over the
    background (and vice versa).  ``signed_coc`` supplies the sign; this
    helper turns it into two non-negative coverage values in ``[0, 1]``.
    """
    max_coc = abs(float(max_coc))
    if max_coc <= _EPS:
        return (0.0, 0.0)

    coc = float(coc)
    if coc < 0.0:
        return (min(1.0, -coc / max_coc), 0.0)
    return (0.0, min(1.0, coc / max_coc))


def composite_near_far(
    sharp_color: Color,
    near_blurred_color: Color,
    far_blurred_color: Color,
    coc: float,
    max_coc: float,
    *,
    near_scale: float = 1.0,
    far_scale: float = 1.0,
) -> tuple[float, ...]:
    """Composite the correct near *or* far blur buffer for a signed CoC.

    ``near_scale`` and ``far_scale`` let an application control the relative
    strength of the two fields without changing their separation.  Because
    only one side is selected per pixel, a background sample can never be
    mixed into a foreground pixel.
    """
    near, far = split_near_far(coc, max_coc)
    if near > 0.0:
        return composite(sharp_color, near_blurred_color, near * max_coc * near_scale, max_coc)
    if far > 0.0:
        return composite(sharp_color, far_blurred_color, far * max_coc * far_scale, max_coc)
    return _color(sharp_color)


# ---------------------------------------------------------------------------
# Transparent geometry / half-resolution diffusion
# ---------------------------------------------------------------------------

def depth_proxy(
    coverage: float,
    particle_depth: float,
    scene_depth: float = _INFINITY,
) -> float:
    """Return an effective depth for a transparent particle fragment.

    Transparent particles normally do not write depth, so a depth-based DOF
    pass sees only the opaque surface behind them and never blurs the
    particles.  Writing a depth proxy makes them participate in DOF while
    still avoiding hard halos at soft particle edges: the particle depth is
    weighted by its alpha coverage and blended with the scene depth behind it.

    Args:
        coverage: Particle alpha / coverage in ``[0, 1]``.
        particle_depth: Depth at which the particle is rendered.
        scene_depth: Opaque scene depth behind the particle.  ``inf`` means
            there is no opaque geometry to blend with.
    """
    coverage = min(1.0, max(0.0, float(coverage)))
    particle_depth = abs(float(particle_depth))
    scene_depth = abs(float(scene_depth))

    if coverage >= 1.0 or math.isinf(scene_depth):
        return particle_depth
    if coverage <= 0.0:
        return scene_depth
    return coverage * particle_depth + (1.0 - coverage) * scene_depth


def half_resolution_coc(coc: float, scale: float = 0.5, max_coc: float | None = None) -> float:
    """Convert a full-resolution CoC radius to half-resolution pixels.

    Per-depth CoC diffusion is performed on a half-resolution buffer.  Using
    the full-resolution radius there doubles the effective blur and produces
    the hard fringe seen around hair edges.  The optional ``max_coc`` clamps
    the value to the pass's maximum radius.
    """
    radius = abs(float(coc))
    if max_coc is not None:
        radius = min(radius, abs(float(max_coc)))
    return radius * abs(float(scale))


# ---------------------------------------------------------------------------
# Autofocus
# ---------------------------------------------------------------------------

def autofocus(
    current_focus: float,
    target_distance: float,
    dt: float,
    speed: float,
    damping: float = 1.0,
    hysteresis: float = 0.0,
) -> float:
    """Move the focus distance toward a target with damping and hysteresis.

    Args:
        current_focus: Previous focus distance.
        target_distance: Distance sampled from the AF point.
        dt: Frame delta time in seconds.  A non-positive ``dt`` holds focus.
        speed: Maximum focus travel in distance units per second.
        damping: Fraction of the available travel applied this frame, in
            ``[0, 1]``.  ``1`` is the ideal velocity-limited tracker; lower
            values smooth out jitter when the camera moves.
        hysteresis: Dead band around the target.  Changes smaller than this do
            not move focus, preventing the lens from hunting back and forth
            between two subjects.

    The move never overshoots the target and is clamped to the segment between
    ``current_focus`` and ``target_distance``.
    """
    current = float(current_focus)
    target = float(target_distance)
    dt = float(dt)
    speed = max(0.0, float(speed))
    damping = min(1.0, max(0.0, float(damping)))
    hysteresis = max(0.0, float(hysteresis))

    error = target - current
    if dt <= 0.0 or abs(error) <= hysteresis or speed <= 0.0 or damping <= 0.0:
        return current

    step = speed * dt * damping
    if step >= abs(error):
        return target

    return current + math.copysign(step, error)
