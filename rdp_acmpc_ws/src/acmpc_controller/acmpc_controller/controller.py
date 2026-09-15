"""The ACMPC control core (§9.11), independent of ROS.

Modes: ``acmpc`` (learned cost map), ``nmpc1`` and ``pid`` -- the realistic
incumbents of §9.8, run on S0/S1 so the deployment result connects to the
study's ledger.

The controller imports the **study module** for the model and the solver rather
than carrying a second implementation.  That is deliberate: ``check_glue``
asserts one set of constants, and a duplicated dynamics model would be a second
thing to keep in step with no test able to see the divergence.  The *estimator*
is the part that must not depend on JAX, and it does not -- it is pure NumPy
``rdp_infer``.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_STUDY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "..", "..", "study")
if os.path.isdir(_STUDY) and _STUDY not in sys.path:
    sys.path.insert(0, os.path.abspath(_STUDY))

from . import bd_bridge                                    # noqa: E402


class ACMPCController:
    """One 50 Hz control step: state + reference + d_hat -> CTBR.

    ``d_hat`` arrives as a **wrench** [N, N.m] and is converted by :mod:`bd_bridge`
    before it reaches the prediction dynamics.  Handing the raw wrench to the
    model is the §4.4 error: a factor m too large in force, dimensionally
    unrelated in moment, and silent.
    """

    def __init__(self, mode="acmpc", horizon=10, n_iter=10, ckpt=None,
                 dmod_mode="first_order", use_d=True):
        import jax.numpy as jnp
        import x500_core_jax as X
        self.X, self.jnp = X, jnp
        self.mode = mode
        self.N = int(horizon)
        self.n_iter = int(n_iter)
        self.dmod_mode = dmod_mode
        self.use_d = bool(use_d)
        self.actor = None
        if mode == "acmpc":
            ck = (X.load_ckpt("acmpc_adaptive", "variants", "C.pkl")
                  or X.load_ckpt("acmpc", "model.pkl")) if ckpt is None else ckpt
            if ck is None:
                raise FileNotFoundError(
                    "no AC-MPC checkpoint found; run Notebooks 3 and 5, or pass "
                    "mode='nmpc1' to fly the incumbent")
            self.actor = ck["actor"]
            self.cfg = dict(ck["cfg"], N=self.N, n_iter=self.n_iter, n_diff=1)
        elif mode == "pid":
            self.pid = X.make_pid_ctrl()
        elif mode != "nmpc1":
            raise ValueError(f"unknown mode {mode!r}")
        self.last_u = np.array([X.U_HOVER, 0.0, 0.0, 0.0])
        self.solve_ms = []

    def step(self, x10, xr_seq, uref_seq, obs=None, d_hat=None):
        """x10 = [p, v, q] ENU/FLU; xr_seq (N+1,10); uref_seq (N,4).

        Returns ``(u_ctbr (4,), info)``.
        """
        jnp, X = self.jnp, self.X
        t0 = time.perf_counter()
        e = np.asarray(X.err(jnp.asarray(x10)[None], jnp.asarray(xr_seq[0])[None]))
        d = None
        if self.use_d and d_hat is not None:
            d = jnp.asarray(bd_bridge.wrench_to_dmod(np.asarray(d_hat),
                                                     self.dmod_mode))[None]
        xs = jnp.asarray(xr_seq)[None]
        us = jnp.asarray(uref_seq)[None]
        if self.mode == "acmpc":
            o = jnp.asarray(obs)[None] if obs is not None else jnp.zeros((1, X.OBS_DIM))
            u, _ = X.mpc_layer(o, jnp.asarray(e), xs, self.actor, self.cfg, d, us)
        elif self.mode == "nmpc1":
            Pt = jnp.broadcast_to(X.PTt, (1, X.NE, X.NE))
            u = X.solve_quad(jnp.asarray(e), xs, self.N, Pt, d, None, self.n_iter,
                             uref_seq=us)
        else:
            u, _ = self.pid(None, jnp.asarray(e), jnp.asarray(xr_seq[0])[None],
                            uref=us[:, 0])
        u = np.asarray(u)[0]
        ms = 1e3 * (time.perf_counter() - t0)
        self.solve_ms.append(ms)
        self.last_u = u
        return u, dict(solve_ms=ms, e=e[0], d_used=None if d is None else np.asarray(d)[0])
