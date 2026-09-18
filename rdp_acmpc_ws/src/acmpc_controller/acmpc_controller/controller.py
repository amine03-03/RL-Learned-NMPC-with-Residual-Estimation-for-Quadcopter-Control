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

    The control model is the 17-state one: ``x = [p, v, q, om, Omega]``.  The
    caller supplies the first thirteen from the EKF; ``Omega`` is NOT telemetered
    by PX4, so this class carries its own estimate -- see :meth:`_with_omega`.
    """

    def __init__(self, mode="acmpc", horizon=10, n_iter=10, ckpt=None,
                 dmod_mode="exact", use_d=True):
        import jax
        import jax.numpy as jnp
        import x500_core_jax as X
        self.X, self.jnp, self.jax = X, jnp, jax
        self._solvers = {}
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
        self.Om_hat = None
        self.reset()

    def reset(self):
        """Cold-start the rotor observer at the hover trim."""
        self.Om_hat = np.full(4, float(self.X.OM_HOVER))

    def _solver(self, no_d):
        """The jitted solve, compiled once per (mode, horizon, d-present).

        Without this the whole iLQR is re-traced on EVERY control step and the
        node cannot hold 50 Hz: measured 5-6 s per call untraced against 1.95 ms
        (N=1, n_iter=5) and 11.45 ms (N=10, n_iter=10) once compiled, on CPU,
        against a 20 ms period.  The first call still pays the compile.
        """
        key = (self.mode, self.N, self.n_iter, bool(no_d))
        if key not in self._solvers:
            jnp, X, jax = self.jnp, self.X, self.jax
            if self.mode == "acmpc":
                actor, cfg = self.actor, self.cfg

                def f(e, xs, us, d, o):
                    return X.mpc_layer(o, e, xs, actor, cfg, d, us)[0]
            else:                                              # 'nmpc1'
                N, n_iter = self.N, self.n_iter

                def f(e, xs, us, d, o):
                    Pt = jnp.broadcast_to(X.PTt, (1, X.NE, X.NE))
                    return X.solve_quad(e, xs, N, Pt, d, None, n_iter, uref_seq=us)

            # d is traced when present and closed over as None when it is not,
            # which is why the cache key carries `no_d`
            self._solvers[key] = (jax.jit(lambda e, xs, us, d, o: f(e, xs, us, None, o))
                                  if no_d else jax.jit(f))
        return self._solvers[key]

    def _with_omega(self, x13):
        """[p, v, q, om] (13,) -> the full 17-state control state.

        PX4 publishes no rotor speed, so Omega is propagated open-loop from the
        commands this controller has already sent, through the SAME rotor model
        the prediction uses (``step_c``).  The mode is stable and fast
        (tau_up = 12.5 ms against dt_c = 20 ms), so the estimate pulls in from a
        cold hover start in about four control steps: measured 0.65 % of
        OM_HOVER on a gentle circle, 2.5-7 % on an aggressive figure of eight,
        where the plant's rate integrator -- which fc() deliberately drops --
        moves Omega_cmd.  Even at the cold-start error the 17-state prediction
        beat the 13-state one, so a coarse Omega is worth more than none.
        """
        x13 = np.asarray(x13, dtype=float).reshape(-1)
        if x13.size == 17:                      # caller supplied Omega itself
            return x13
        if x13.size != 13:
            raise ValueError(
                f"expected a 13-element [p,v,q,om] state (or 17 with Omega), got {x13.size}")
        return np.concatenate([x13, self.Om_hat])

    def step(self, x13, xr_seq, uref_seq, obs=None, d_hat=None):
        """x13 = [p, v, q, om] ENU/FLU; xr_seq (N+1,17); uref_seq (N,4).

        ``xr_seq`` comes from :mod:`refgen`, which mirrors the study's
        ``ref_state``.  Returns ``(u_ctbr (4,), info)``.
        """
        jnp, X = self.jnp, self.X
        t0 = time.perf_counter()
        x17 = self._with_omega(x13)
        xr_seq = np.asarray(xr_seq, dtype=float)
        if xr_seq.shape[-1] != X.NX:
            raise ValueError(
                f"xr_seq has width {xr_seq.shape[-1]}, expected NX = {X.NX}; "
                "build it with refgen.ref_traj")
        e = np.asarray(X.err(jnp.asarray(x17)[None], jnp.asarray(xr_seq[0])[None]))
        d = None
        if self.use_d and d_hat is not None:
            d = jnp.asarray(bd_bridge.wrench_to_dmod(np.asarray(d_hat),
                                                     self.dmod_mode))[None]
        xs = jnp.asarray(xr_seq)[None]
        us = jnp.asarray(uref_seq)[None]
        if self.mode == "pid":
            u, _ = self.pid(None, jnp.asarray(e), jnp.asarray(xr_seq[0])[None],
                            uref=us[:, 0])
        else:
            o = (jnp.asarray(obs)[None] if obs is not None
                 else jnp.zeros((1, X.OBS_DIM)))
            u = self._solver(d is None)(jnp.asarray(e), xs, us, d, o)
        u = np.asarray(u)[0]
        # advance the rotor observer with the command actually issued, through
        # the same model the solver predicted with
        self.Om_hat = np.asarray(
            X.step_c(jnp.asarray(x17)[None], jnp.asarray(u)[None]))[0, 13:17]
        ms = 1e3 * (time.perf_counter() - t0)
        self.solve_ms.append(ms)
        self.last_u = u
        return u, dict(solve_ms=ms, e=e[0], d_used=None if d is None else np.asarray(d)[0],
                       Om_hat=self.Om_hat.copy())
