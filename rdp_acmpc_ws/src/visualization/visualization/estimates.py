"""Turn the RDP's 6-D residual wrench into the quantities a payload demo shows.

The predictor outputs ``[F_ext (3, world ENU, N), tau_ext (3, body FLU, N m)]``:
the residual of the **nominal-mass, nominal-inertia** model (4.10).  It does not
output a mass, and pretending otherwise would be the interesting kind of wrong.
What it does output is enough to *derive* one, under a stated assumption, and
this module keeps that derivation in one place with the assumption attached.

Mass.  With true mass ``m = m_n + m_p`` the residual force is exactly

    F = -T z_b (m_p / m)                                              (9.9)

and in hover ``T z_b -> m g e3``, so ``F_z -> -m_p g`` and

    m_hat = m_n - F_z / g                                             (9.10)

:func:`mass_estimate` uses (9.10) by default -- the honest hover form, one
division, no extra inputs.  When the thrust and attitude are on hand it can
instead invert (9.9) exactly, which stays right through the tilt transient;
pass ``thrust`` and ``q``.  Tilt alone does not separate them: in a steady tilted hold
``T cos(tilt) = m g`` exactly, so (9.10) is exact at any tilt.  What separates
them is **vertical acceleration** -- ``T cos(tilt) = m (g + a_z)`` makes (9.10)
read ``m_p (g + a_z) / g``, biased by +10.2 % per 1 m/s^2 of climb.  A hover
hold is the benign case and the exact branch is the one to want during the
take-off transient, which is why the mode is recorded on the plot rather than
chosen silently.

Moment.  ``tau_ext`` **is** the moment estimate; there is nothing to derive.
What is worth deriving is the CG offset it implies, since that is the physical
claim the demonstration makes:

    tau = r_p x (R^T F),  hover:  tau = (-m_p g r_y, +m_p g r_x, 0)

    r_x_hat = +tau_y / (m_p_hat g),   r_y_hat = -tau_x / (m_p_hat g)  (9.11)

:func:`cg_offset_estimate` returns that, and returns NaN rather than a large
number when ``m_p_hat`` is too small to divide by -- an unbounded offset drawn
from a near-zero payload is a plotting artefact, not an estimate.
"""
from __future__ import annotations

import numpy as np

#: Asserted equal to the study's by ``check_glue``.  Not a second source of truth.
M_NOM = 2.0643076923076924
G = 9.8066

#: Below this estimated payload the CG offset is unidentifiable: the moment and
#: the force both go to zero and their ratio is noise over noise.  30 g is 1.5 %
#: of vehicle mass and about three times the RDP's own hover-force scatter.
MIN_PAYLOAD_KG = 0.030


def qrotmat(q):
    """Scalar-first quaternion -> rotation matrix (body -> world)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def mass_estimate(F_ext, thrust=None, q=None, m_nom=M_NOM, g=G):
    """-> (m_hat, m_p_hat) in kg, from the residual force.

    ``F_ext`` is (..., 3) in world ENU.  With ``thrust`` (total, N) and ``q``
    (attitude, scalar-first ENU/FLU) this inverts (9.9) exactly::

        F . z_b = -T (m_p / m)   =>   m_p / m = -(F . z_b) / T

    Without them it uses the hover form (9.10).  The exact branch falls back to
    the hover form wherever the thrust is too small to divide by, so a logged
    run with a few zero-thrust frames does not produce infinities.
    """
    F = np.asarray(F_ext, float)
    Fz = F[..., 2]
    if thrust is None or q is None:
        m_p = -Fz / g
        return m_nom + m_p, m_p
    T = np.asarray(thrust, float)
    zb = qrotmat(np.asarray(q, float))[..., :, 2]          # body z in world
    proj = np.einsum("...i,...i->...", F, zb)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.abs(T) > 1e-6, -proj / np.maximum(T, 1e-6), np.nan)
        # m_p / m = frac  =>  m_p = m_n frac / (1 - frac)
        m_p = np.where(np.abs(1.0 - frac) > 1e-6, m_nom * frac / (1.0 - frac), np.nan)
    m_p = np.where(np.isfinite(m_p), m_p, -Fz / g)         # hover fallback
    return m_nom + m_p, m_p


def cg_offset_estimate(tau_ext, m_p_hat, g=G, min_payload=MIN_PAYLOAD_KG):
    """-> (r_x_hat, r_y_hat) in m, by (9.11).  NaN when the payload is too small.

    Only the in-plane components are identifiable: a payload directly below the
    CG makes no moment at hover, so ``r_z`` does not appear in (9.11) and is not
    returned.  Reporting a third number the data cannot support would be worse
    than reporting two.
    """
    tau = np.asarray(tau_ext, float)
    m_p = np.asarray(m_p_hat, float)
    ok = np.abs(m_p) >= float(min_payload)
    with np.errstate(divide="ignore", invalid="ignore"):
        rx = np.where(ok, +tau[..., 1] / (m_p * g), np.nan)
        ry = np.where(ok, -tau[..., 0] / (m_p * g), np.nan)
    return rx, ry


def payload_truth(m_p, r_p, m_nom=M_NOM, g=G):
    """The ground-truth lines to draw beside the estimates, from S7's parameters."""
    r = np.asarray(r_p, float)
    return dict(m_total=m_nom + float(m_p), m_p=float(m_p),
                F_z=-float(m_p) * g,
                tau=np.array([-m_p * g * r[1], +m_p * g * r[0], 0.0]),
                r_x=float(r[0]), r_y=float(r[1]))


def settling_time(t, y, target, tol, hold_s=0.5):
    """First time after which ``|y - target| <= tol`` for at least ``hold_s``.

    NaN when it never settles.  Used for the adaptation time annotated on the
    mass panel; the threshold is a parameter and is recorded, because a settling
    time quoted without its tolerance is not a number.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    if t.size < 2:
        return float("nan")
    inside = np.abs(y - float(target)) <= float(tol)
    dt = float(np.median(np.diff(t))) or 1.0
    need = max(int(round(float(hold_s) / dt)), 1)
    run = 0
    for i, ok in enumerate(inside):
        run = run + 1 if ok else 0
        if run >= need and bool(inside[i:].all()):
            return float(t[i - need + 1])
    return float("nan")
