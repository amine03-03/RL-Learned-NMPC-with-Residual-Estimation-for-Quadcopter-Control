"""Reference trajectory of §9.9 and its feasibility check.

    x_d = A cos(w t),   y_d = B sin(2 w t),   z_d = z0                   (9.5)

Three aggressiveness levels by changing w.  **Each is checked against the
feasibility envelope (2.6) before it is flown**: peak demand is
max_t ||p_d_ddot|| and must stay under alpha * a_lat_max = 8.0098 m/s^2.

Pure NumPy; importable with no ROS on the path.
"""
from __future__ import annotations

import numpy as np

#: Derived in the study module and asserted equal by ``check_glue`` -- do not
#: re-derive them here.  A second copy of a constant is a second thing to keep
#: in step, and §9.4 check 3 is about exactly that failure.
A_LAT_MAX = 13.34971104560659
ALPHA = 0.60
A_LAT_BUDGET = ALPHA * A_LAT_MAX

LEVELS = {"gentle": 0.60, "moderate": 1.00, "aggressive": 1.45}   # rad/s


def lissajous(t, A=1.2, B=0.9, w=1.0, z0=1.5):
    """Position, velocity and acceleration of (9.5), exact analytic derivatives."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    p = np.stack([A * np.cos(w * t), B * np.sin(2 * w * t), np.full_like(t, z0)], -1)
    v = np.stack([-A * w * np.sin(w * t), 2 * B * w * np.cos(2 * w * t),
                  np.zeros_like(t)], -1)
    a = np.stack([-A * w ** 2 * np.cos(w * t), -4 * B * w ** 2 * np.sin(2 * w * t),
                  np.zeros_like(t)], -1)
    return p, v, a


def peak_demand(A=1.2, B=0.9, w=1.0, n=200_001):
    """max_t ||p_d_ddot||, by dense sampling of one period."""
    t = np.linspace(0.0, 2 * np.pi / w, n)
    _, v, a = lissajous(t, A, B, w)
    return float(np.linalg.norm(v, axis=1).max()), float(np.linalg.norm(a, axis=1).max())


def check_feasible(A=1.2, B=0.9, w=1.0, budget=A_LAT_BUDGET):
    """Return (ok, peak_v, peak_a, budget).  **Call before flying.**"""
    pv, pa = peak_demand(A, B, w)
    return bool(pa <= budget + 1e-9), pv, pa, budget


def largest_feasible_w(A=1.2, B=0.9, budget=A_LAT_BUDGET, hi=4.0):
    """Bisect for the fastest feasible w at this amplitude."""
    lo = 1e-3
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if peak_demand(A, B, mid)[1] <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def episode_timeline(t):
    """§9.9: 0-5 s nominal, 5-12 s disturbance active, 12-15 s recovery.

    Returned as a label so the logger and the plots agree on the phase
    boundaries instead of each hard-coding them.
    """
    t = float(t)
    if t < 5.0:
        return "nominal"
    if t < 12.0:
        return "disturbed"
    return "recovery"


DISTURBANCE_ON, DISTURBANCE_OFF, EPISODE_END = 5.0, 12.0, 15.0
