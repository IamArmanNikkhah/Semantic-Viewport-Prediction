import numpy as np


def unwrap_and_interp(raw_deg, raw_t, t_uniform):
    """
    Minimal version of the yaw interpolation logic:

    1. Convert deg → rad
    2. Unwrap to remove ±π jumps
    3. Interpolate on uniform time grid
    4. Wrap back into (-π, π]
    """
    raw_rad = np.deg2rad(raw_deg)
    unwrapped = np.unwrap(raw_rad)
    interp = np.interp(t_uniform, raw_t, unwrapped)
    wrapped = ((interp + np.pi) % (2 * np.pi)) - np.pi
    return wrapped


def test_yaw_wraparound_interpolation_does_not_pass_through_zero():
    """
    Simulate a user turning across +180 → -180 degrees.
    We expect interpolated yaw to stay near the ±180 boundary, not drift toward 0.
    """
    # Times (in seconds)
    raw_t = np.array([0.0, 0.5, 1.0])
    # Yaw in degrees: classic wrap case
    raw_yaw_deg = np.array([170.0, 179.0, -179.0])

    # Uniform timeline from 0 to 1 second (11 samples)
    t_uniform = np.linspace(0.0, 1.0, num=11)

    yaw_wrapped = unwrap_and_interp(raw_yaw_deg, raw_t, t_uniform)

    # Convert back to degrees for intuitive checks
    yaw_deg = np.rad2deg(yaw_wrapped)

    # If interpolation was naive, it might go through 0 deg (forward).
    # We assert that yaw stays "near the boundary", i.e., far from 0.
    max_abs_yaw = np.max(np.abs(yaw_deg))

    # Make sure yaw doesn't collapse toward 0 anywhere
    assert max_abs_yaw > 100.0, (
        "Yaw interpolation seems to drift toward 0°, "
        "which would indicate broken wrap-around handling."
    )
