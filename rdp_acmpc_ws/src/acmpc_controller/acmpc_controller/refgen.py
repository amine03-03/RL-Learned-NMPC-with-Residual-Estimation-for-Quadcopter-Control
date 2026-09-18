"""Build the 17-state reference the control model needs, in pure NumPy.

The study's ``x500_core_jax.ref_state`` does this for the six analytic paths of
§5.6.  The node flies a Lissajous, which is not one of them, so the *conversion*
-- (p, v, a) -> (p, v, q, om, Omega) + u_ref -- is reproduced here and checked
against the study's by ``check_glue``.  The differencing scheme is identical:
om_ref from a centred difference of q_ref at +/- dt_c/2, om_dot_ref from a
centred difference of that.
"""
from __future__ import annotations

import numpy as np

from . import frames as F

G = 9.8066
M_NOM = 2.0643076923076924
K_T = 8.54858e-06
OM_ROTOR_MIN, OM_ROTOR_MAX = 150.0, 1000.0
T_MAX = 4.0 * K_T * OM_ROTOR_MAX ** 2
F_MAX = K_T * OM_ROTOR_MAX ** 2
OM_MAX = np.array([10.0, 10.0, 4.0])
J_NOM = np.diag([0.023832493376032816, 0.02393541818638026, 0.04399995371400395])
#: nominal allocator inverse (2.7) with the PX4 k_m = 0.05 belief (A3)
_ARMS = np.array([[+0.174, -0.174, 0.06], [-0.174, +0.174, 0.06],
                  [+0.174, +0.174, 0.06], [-0.174, -0.174, 0.06]])
_SIGMA = np.array([+1.0, +1.0, -1.0, -1.0])
_CG = np.array([0.0, 0.0, (4 * 0.016076923076923075 * 0.06) / M_NOM])
_A = _ARMS - _CG
MINV_CTRL = np.linalg.inv(np.vstack([np.ones(4), _A[:, 1], -_A[:, 0], -_SIGMA * 0.05]))


def ref_attitude(a):
    """(4.7): the zero-yaw attitude whose body z axis carries a + g e3."""
    zb = np.asarray(a, float) + np.array([0.0, 0.0, G])
    n = np.linalg.norm(zb)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    zb = zb / n
    ax = np.cross(np.array([0.0, 0.0, 1.0]), zb)
    s = np.linalg.norm(ax)
    ang = np.arccos(np.clip(zb[2], -1.0, 1.0))
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[np.cos(ang / 2)], ax / s * np.sin(ang / 2)])


def _om_at(pva, t, h):
    """Reference body rate at t, by the centred difference of §5.6."""
    q = ref_attitude(pva(t)[2])
    qd = (ref_attitude(pva(t + h)[2]) - ref_attitude(pva(t - h)[2])) / (2 * h)
    return 2.0 * F.qmul(F.qconj(q), qd)[1:4]


def ref_stage(pva, t, dt_c=0.02):
    """One stage: -> (xr (17,), u_ref (4,)).

    ``pva(t)`` must return ``(p, v, a)`` as three length-3 arrays.
    """
    p, v, a = (np.asarray(z, float).reshape(3) for z in pva(t))
    h = 0.5 * dt_c
    q = ref_attitude(a)
    om = _om_at(pva, t, h)
    omdot = (_om_at(pva, t + h, h) - _om_at(pva, t - h, h)) / (2 * h)

    # (4.8): exact inversion of (2.10) INCLUDING the idle floor (A2)
    T_ref = float(np.clip(M_NOM * np.linalg.norm(a + np.array([0.0, 0.0, G])),
                          0.0, T_MAX))
    r = OM_ROTOR_MIN / OM_ROTOR_MAX
    c = float(np.clip((np.sqrt(T_ref / T_MAX) - r) / (1.0 - r), 0.0, 1.0))
    u_ref = np.concatenate([[c], om / OM_MAX])

    tau = J_NOM @ omdot + np.cross(om, J_NOM @ om)
    f = np.clip(MINV_CTRL @ np.concatenate([[T_ref], tau]), 0.0, F_MAX)
    Om = np.clip(np.sqrt(f / K_T), OM_ROTOR_MIN, OM_ROTOR_MAX)
    return np.concatenate([p, v, q, om, Om]), u_ref


def ref_traj(pva, t0, N, dt):
    """-> (xr (N+1,17), u_ref (N,4)) over the preview horizon."""
    xs, us = [], []
    for k in range(N + 1):
        x, u = ref_stage(pva, t0 + k * dt)
        xs.append(x)
        if k < N:
            us.append(u)
    return np.stack(xs), np.stack(us)
