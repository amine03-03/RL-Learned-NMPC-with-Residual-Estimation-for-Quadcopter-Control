"""B_d: the wrench -> model-residual conversion of §4.4, **implemented once**.

§9.1 requires that B_d be "exactly the conversion of §4.4: force -> acceleration
via 1/m, moment -> commanded rate via (4.13).  Implement it once, in one
function, shared by study and node."  This module is that one place for the
workspace; ``check_glue`` asserts it agrees with the study's
``wrench_to_dmod`` to 1e-12 on random wrenches.

Getting it wrong is silent.  Feeding the raw wrench into the model is a factor
m = 2.064 too large in force and dimensionally unrelated in moment, and it
raises no exception -- it just produces plausible numbers that are wrong in the
same direction every time.
"""
from __future__ import annotations

import numpy as np

M_NOM = 2.0643076923076924
J_NOM = np.diag([0.023832493376032816, 0.02393541818638026, 0.04399995371400395])
K_RATE = np.array([14.0, 14.0, 8.0])

#: (4.13): om_res ~ diag(K_r)^-1 J^-1 tau.  A *model*, valid while the horizon is
#: short against the integral time constant (1/K_r = 71 ms vs 1/K_i = 250 ms).
_MOMENT_GAIN = np.linalg.inv(np.diag(K_RATE) @ J_NOM)


#: Exact closed-loop gain of the rate loop to a step moment, at
#: ``DMOD_SETTLE_S``.  (4.13) neglects K_i, so it is the t -> 0 limit; measured,
#: the true rate residual is ~0.9 of it out to 0.3 s but 0.10 by 8 s, i.e. a
#: standing moment is overstated ~9x (D3).  Mirrors ``x500_core_jax.moment_gain``
#: and is checked against it by ``check_glue``.
DMOD_SETTLE_S = 0.5


def moment_gain(t):
    Kr, Ki = K_RATE, np.array([4.0, 4.0, 2.0])
    disc = np.sqrt(np.maximum(Kr ** 2 - 4 * Ki, 1e-300))
    rp, rm = (-Kr + disc) / 2, (-Kr - disc) / 2
    return Kr * (np.exp(rp * t) - np.exp(rm * t)) / (rp - rm)


def wrench_to_dmod(w, mode="first_order"):
    """[F (world, N), tau (body, N.m)] -> [a_res (m/s^2), om_res (rad/s)].

    ``mode='closed_loop'`` scales the moment block by :func:`moment_gain`;
    ``'none'`` zeroes it, which is the correct limit once the rate integrator
    has absorbed a standing moment.
    """
    w = np.asarray(w, dtype=float)
    single = w.ndim == 1
    w = np.atleast_2d(w)
    a_res = w[:, 0:3] / M_NOM                                        # (4.12)
    if mode == "none":
        om_res = np.zeros_like(a_res)
    elif mode in ("first_order", "closed_loop"):
        om_res = w[:, 3:6] @ _MOMENT_GAIN.T                           # (4.13)
        if mode == "closed_loop":
            om_res = om_res * moment_gain(DMOD_SETTLE_S)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    out = np.concatenate([a_res, om_res], -1)
    return out[0] if single else out


def worked_example():
    """The §4.4 example, so a regression shows up as a number and not a crash."""
    d = wrench_to_dmod(np.array([6.478, 0, 0, 0, 0, 0]))
    d2 = wrench_to_dmod(np.array([0, 0, 0, 0.3452, 0, 0]))
    return dict(a_res=float(d[0]), om_res=float(d2[3]),
                a_res_expected=3.138, om_res_expected=1.035)
