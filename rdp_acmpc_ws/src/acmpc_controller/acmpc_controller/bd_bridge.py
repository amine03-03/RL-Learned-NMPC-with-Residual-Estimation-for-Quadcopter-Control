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
J_NOM = np.diag([0.023830954779613455, 0.023830954779613455, 0.043893959384615384])
K_RATE = np.array([14.0, 14.0, 8.0])

#: (4.13): om_res ~ diag(K_r)^-1 J^-1 tau.  A *model*, valid while the horizon is
#: short against the integral time constant (1/K_r = 71 ms vs 1/K_i = 250 ms).
_MOMENT_GAIN = np.linalg.inv(np.diag(K_RATE) @ J_NOM)


def wrench_to_dmod(w, mode="first_order"):
    """[F (world, N), tau (body, N.m)] -> [a_res (m/s^2), om_res (rad/s)]."""
    w = np.asarray(w, dtype=float)
    single = w.ndim == 1
    w = np.atleast_2d(w)
    a_res = w[:, 0:3] / M_NOM                                        # (4.12)
    if mode == "none":
        om_res = np.zeros_like(a_res)
    elif mode == "first_order":
        om_res = w[:, 3:6] @ _MOMENT_GAIN.T                           # (4.13)
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
