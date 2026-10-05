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

from . import _study_path                                 # noqa: E402,F401
from . import bd_bridge                                    # noqa: E402


class ACMPCController:
    """One 50 Hz control step: state + reference + d_hat -> CTBR.

    ``d_hat`` arrives as a **wrench** [N, N.m] and is converted by :mod:`bd_bridge`
    before it reaches the prediction dynamics.  Handing the raw wrench to the
    model is the §4.4 error: a factor m too large in force, dimensionally
    unrelated in moment, and silent.
    """

    def __init__(self, mode="acmpc", horizon=10, n_iter=10, ckpt=None,
                 dmod_mode="first_order", use_d=True, delay_steps=0):
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
            # variant C was trained with the 6-D wrench channel appended to its
            # observation (EnvCfg.oracle, target 'wrench'); A/B and the nominal
            # AC-MPC were not
            self.oracle = bool(ck.get("to_obs", False))
        elif mode == "pid":
            self.pid = X.make_pid_ctrl()
        elif mode != "nmpc1":
            raise ValueError(f"unknown mode {mode!r}")
        self.oracle = getattr(self, "oracle", False)
        # Delay compensation.  The study applies each command at the step it was
        # computed; a real loop (odometry transport, the solve, DDS, PX4) applies
        # it about one control period later, and the NMPC tuned without delay
        # diverges on the study plant with a single 20 ms step of it.  The state
        # is therefore propagated through the in-flight commands with the
        # controller's own model (step_c) and the problem solved from there.
        self.delay_steps = int(delay_steps)
        self._inflight = [np.array([X.U_HOVER, 0.0, 0.0, 0.0])] * self.delay_steps
        import jax as _jax
        self._step_c = _jax.jit(lambda x, u, d: X.step_c(x, u, d))
        # jit each solve ONCE (as make_nmpc_ctrl does in the study): traced
        # per call, one N=10 iLQR solve costs ~6 s instead of ~15 ms
        import jax
        if mode == "nmpc1":
            Pt = jnp.broadcast_to(X.PTt, (1, X.NE, X.NE))
            N, it = self.N, self.n_iter
            self._solve = jax.jit(lambda o, e, xs, us, d: X.solve_quad(
                e, xs, N, Pt, d, None, it, uref_seq=us))
        elif mode == "acmpc":
            actor, cfg = self.actor, self.cfg
            self._solve = jax.jit(lambda o, e, xs, us, d: X.mpc_layer(
                o, e, xs, actor, cfg, d, us)[0])
        self.last_u = np.array([X.U_HOVER, 0.0, 0.0, 0.0])
        self.solve_ms = []
        self.ref = None

    def set_reference(self, pva, hold=False):
        """Attach the analytic reference ``pva(t) -> (p, v, a)`` used by :meth:`tick`."""
        from .reference import ObsBuilder
        self.ref = (pva, bool(hold))
        self.obs_builder = ObsBuilder(pva, hold, u0=self.last_u)
        self.u_ref_prev = np.array([self.X.U_HOVER, 0.0, 0.0, 0.0])
        self.n_tick = 0

    def predict(self, x10, d_hat=None):
        """State after the ``delay_steps`` commands already in flight."""
        if not self.delay_steps:
            return np.asarray(x10, float)
        X, jnp = self.X, self.jnp
        d = None
        if self.use_d:
            d_hat = np.zeros(6) if d_hat is None else np.asarray(d_hat, float)
            d = jnp.asarray(bd_bridge.wrench_to_dmod(d_hat, self.dmod_mode))[None]
        x = jnp.asarray(x10)[None]
        for u in self._inflight:
            x = self._step_c(x, jnp.asarray(u)[None], d)
        return np.asarray(x)[0]

    def tick(self, x10, om, t, d_hat=None):
        """One control period, everything built with the study's functions.

        x10 = [p, v, q] ENU/FLU, om = body rate FLU, t = reference time [s].
        Returns ``(u (4,), info)``; ``u`` is clipped to the study's input box
        and info carries the reference and ``u_prev`` for ``/acmpc/status``.
        """
        from .reference import ref_sequence
        X, jnp = self.X, self.jnp
        if self.ref is None:
            raise RuntimeError("call set_reference() before tick()")
        pva, hold = self.ref
        x_meas = np.asarray(x10, float)
        x10, t = self.predict(x_meas, d_hat), t + self.delay_steps * X.P.dt_c
        ts = t + np.arange(self.N + 1) * X.P.dt_c
        xr, ur = ref_sequence(pva, ts, hold)
        e = np.asarray(X.err(jnp.asarray(x10)[None], jnp.asarray(xr[0])[None]))[0]
        u_prev = self.last_u.copy()
        if self.n_tick > 0:          # at reset the study observes prev as initialised
            self.obs_builder.update(e, om, u_prev - self.u_ref_prev, u_prev)
        self.n_tick += 1
        obs = None
        if self.mode == "acmpc":
            oracle = None
            if self.oracle:
                oracle = np.zeros(6) if d_hat is None else np.asarray(d_hat, float)
            obs = self.obs_builder.obs(e, x10[0:3], t, oracle)
        u, info = self.step(x10, xr, ur[:self.N], obs=obs, d_hat=d_hat)
        u = np.clip(u, X.U_LO, X.U_HI)
        self.last_u = u
        if self.delay_steps:
            self._inflight = self._inflight[1:] + [u.copy()]
        self.u_ref_prev = ur[0]
        info.update(x_ref=xr[0], u_ref=ur[0], u_prev=u_prev, obs=obs)
        return u, info

    def step(self, x10, xr_seq, uref_seq, obs=None, d_hat=None):
        """x10 = [p, v, q] ENU/FLU; xr_seq (N+1,10); uref_seq (N,4).

        Returns ``(u_ctbr (4,), info)``.
        """
        jnp, X = self.jnp, self.X
        t0 = time.perf_counter()
        e = np.asarray(X.err(jnp.asarray(x10)[None], jnp.asarray(xr_seq[0])[None]))
        d = None
        if self.use_d:                   # zeros until an estimate arrives: one trace
            d_hat = np.zeros(6) if d_hat is None else np.asarray(d_hat, float)
            d = jnp.asarray(bd_bridge.wrench_to_dmod(d_hat, self.dmod_mode))[None]
        xs = jnp.asarray(xr_seq)[None]
        us = jnp.asarray(uref_seq)[None]
        if self.mode == "acmpc":
            if obs is None:
                raise ValueError("acmpc needs the (5.3) observation; use tick()")
            u = self._solve(jnp.asarray(obs)[None], jnp.asarray(e), xs, us, d)
        elif self.mode == "nmpc1":
            u = self._solve(None, jnp.asarray(e), xs, us, d)
        else:
            u, _ = self.pid(None, jnp.asarray(e), jnp.asarray(xr_seq[0])[None],
                            uref=us[:, 0])
        u = np.asarray(u)[0]
        ms = 1e3 * (time.perf_counter() - t0)
        self.solve_ms.append(ms)
        self.last_u = u
        return u, dict(solve_ms=ms, e=e[0], d_used=None if d is None else np.asarray(d)[0])
