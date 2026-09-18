"""B_d: the wrench -> model-residual conversion of §4.4, **implemented once**.

§9.1 requires that B_d be "exactly the conversion of §4.4: force -> acceleration
via 1/m, moment -> angular acceleration via J^-1.  Implement it once, in one
function, shared by study and node."  This module is that one place for the
workspace; ``check_glue`` asserts it agrees with the study's ``wrench_to_dmod``
to 1e-12 on random wrenches.

Getting it wrong is silent.  Feeding the raw wrench into the model is a factor
m = 2.064 too large in force and dimensionally unrelated in moment, and it
raises no exception -- it just produces plausible numbers that are wrong in the
same direction every time.

**Both blocks are now exact.**  Under the 10-state control model the moment
block had no exact image -- there was no torque input -- so (4.13) approximated
it by the steady state of the rate loop with K_i neglected.  That model
overstated a standing moment by ~9x at 8 s (D3/D4) and measured 69-103x SMALLER
than the (4.9) residual it was meant to cancel.  The 17-state model carries
omega and Omega, so tau maps to angular acceleration by J_nom^-1 and the
``first_order``/``closed_loop`` modes, ``moment_gain`` and ``DMOD_SETTLE_S``
are gone.
"""
from __future__ import annotations

import numpy as np

M_NOM = 2.0643076923076924
J_NOM = np.diag([0.023832493376032816, 0.02393541818638026, 0.04399995371400395])
JINV_NOM = np.linalg.inv(J_NOM)

#: Mirrors ``x500_core_jax.DMOD_MODES``.
DMOD_MODES = ("exact", "none")


def wrench_to_dmod(w, mode="exact"):
    """[F (world, N), tau (body, N.m)] -> [a_res (m/s^2), alpha_res (rad/s^2)].

    ``mode='none'`` zeroes the moment block; it exists only as an ablation.
    """
    if mode not in DMOD_MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {DMOD_MODES}")
    w = np.asarray(w, dtype=float)
    single = w.ndim == 1
    w = np.atleast_2d(w)
    a_res = w[:, 0:3] / M_NOM                                        # (4.12)
    if mode == "none":
        al_res = np.zeros_like(a_res)
    else:
        al_res = w[:, 3:6] @ JINV_NOM.T
    out = np.concatenate([a_res, al_res], axis=1)
    return out[0] if single else out


def _selftest():                                           # pragma: no cover
    d = wrench_to_dmod(np.array([6.478, 0, 0, 0, 0, 0]))
    assert abs(d[0] - 6.478 / M_NOM) < 1e-12, d
    d2 = wrench_to_dmod(np.array([0, 0, 0, 0.3452, 0, 0]))
    assert abs(d2[3] - 0.3452 / J_NOM[0, 0]) < 1e-12, d2
    assert np.abs(wrench_to_dmod(np.zeros((4, 6)), "none")[:, 3:]).max() == 0.0
    print("bd_bridge selftest ok")


if __name__ == "__main__":                                 # pragma: no cover
    _selftest()
