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

from . import bd_bridge                                    # noqa: E402
from .px4_topics import add_study_to_path                  # noqa: E402

#  Counting `..` from __file__ worked only by coincidence: with
#  `colcon build --symlink-install` the module that runs lives under build/,
#  at a different depth from src/.  Search for the directory instead.
add_study_to_path()


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
            #  Preference order, and the reason for it.  Variant B routes the
            #  residual into the MODEL and leaves the cost map
            #  disturbance-blind; variant C routes it into both.  The scenario
            #  sweep puts B ahead of C everywhere, so B is what this flies when
            #  it is available -- and B needs no oracle observation channel,
            #  which makes its observation the plain OBS_DIM one.  C is kept as
            #  a fallback because it is still an adaptive policy; the plain
            #  Notebook-3 model is the last resort and is not adaptive at all.
            self.ckpt_src = None
            if ckpt is None:
                for src in (("acmpc_adaptive", "variants", "B.pkl"),
                            ("acmpc_adaptive", "variants", "C.pkl"),
                            ("acmpc", "model.pkl")):
                    ck = X.load_ckpt(*src)
                    if ck is not None:
                        self.ckpt_src = "/".join(src)
                        break
            else:
                ck, self.ckpt_src = ckpt, "caller-supplied"
            if ck is None:
                raise FileNotFoundError(
                    "no AC-MPC checkpoint found; run Notebooks 3 and 5, or pass "
                    "mode='nmpc1' to fly the incumbent")
            self.actor = ck["actor"]

            #  The horizon of a learned cost map is NOT a free parameter.  The
            #  head emits one parameter block per stage in a single dense
            #  layer, so its output width is N_train * REP_DIM[rep] and is
            #  fixed at training time.  Overriding N here asked a head trained
            #  for one stage to fill ten, which fails inside the cost map as
            #  "cannot reshape array of shape (1, 40) into (1, 10, 40)" -- a
            #  long way from the horizon parameter that caused it.
            #  n_iter IS free: it only says how long to iterate the solve.
            n_train = int(ck["cfg"].get("N", 1))
            if int(horizon) != n_train:
                print(f"ACMPCController: this checkpoint was trained at "
                      f"N = {n_train}; flying it at N = {horizon} is not "
                      f"possible because the cost-map head has one output "
                      f"block per stage. Using N = {n_train}.", file=sys.stderr)
            self.N = n_train
            self.cfg = dict(ck["cfg"], N=self.N, n_iter=self.n_iter, n_diff=1)

            #  What observation width was this actor TRAINED with?  Read it off
            #  the normaliser rather than assuming X.OBS_DIM: the adaptive
            #  variants that route the residual into the observation as well as
            #  the model (variant C) were trained with the six oracle channels
            #  on, so their actor expects OBS_DIM + 6.  Handing such an actor a
            #  47-wide vector is a broadcasting error inside the cost map --
            #  loud, but a long way from its cause.
            mu = (self.actor.get("obs_norm") or {}).get("mu")
            self.obs_w = int(mu.shape[-1]) if mu is not None else int(X.OBS_DIM)
            self.oracle_n = self.obs_w - int(X.OBS_DIM)
            if self.oracle_n not in (0, 6):
                raise ValueError(
                    f"this checkpoint expects a {self.obs_w}-wide observation, "
                    f"which is neither OBS_DIM ({X.OBS_DIM}) nor OBS_DIM + 6. "
                    f"It was trained against a different observation layout, so "
                    f"flying it here would evaluate the learned cost at points "
                    f"it never saw. Re-export it, or pass mode='nmpc1'.")
            #  'wrench' feeds the raw estimate in [N, N.m]; 'residual' feeds it
            #  converted.  The training config says which, and getting it wrong
            #  is silent -- the widths match either way.
            self.oracle_target = self.cfg.get("oracle_target", "wrench")
        elif mode == "pid":
            self.pid = X.make_pid_ctrl()
        elif mode != "nmpc1":
            raise ValueError(f"unknown mode {mode!r}")
        self.last_u = np.array([X.U_HOVER, 0.0, 0.0, 0.0])
        self.solve_ms = []
        self.Om_hat = None
        #  (5.3) carries memory: an integral of the position error and three
        #  exponential means.  The training environment keeps it per vehicle
        #  and advances it every step; a controller that does not is handing
        #  the cost map four blocks of zeros for the whole flight.
        self._ema_a = float(np.exp(-X.P.dt_c / X.EMA_TAU))
        self._warned_preview = False
        self.reset()

    def reset(self):
        """Cold-start the rotor observer and the observation memory."""
        self.Om_hat = np.full(4, float(self.X.OM_HOVER))
        self.int_ep = np.zeros(3)
        self.du_bar = np.zeros(4)
        self.ev_bar = np.zeros(3)
        self.om_bar = np.zeros(3)
        self.last_u = np.array([self.X.U_HOVER, 0.0, 0.0, 0.0])

    def _build_obs(self, e, x17, xr_seq, preview, d_hat):
        """(5.3): the 47 channels the cost map was trained on, plus the oracle.

            e (16) | 3 x [p_ref(t+h) - p, v_ref(t+h)] (18) | int_ep (3)
                   | du_bar (4) | ev_bar (3) | om_bar (3)

        ``preview`` is (3, 6) built by the caller from the reference it is
        flying.  Under a position hold every lookahead is the same fixed
        setpoint, so it can be derived from ``xr_seq[0]`` -- but only under a
        hold, and a tracking reference that relies on that silently trains the
        cost map on a preview that does not move.  So the fallback says so
        once.
        """
        X, np_ = self.X, np
        p = np_.asarray(x17[0:3], float)
        if preview is None:
            if not self._warned_preview:
                self._warned_preview = True
                print("ACMPCController: no reference preview supplied; assuming "
                      "a position hold, where every lookahead is the same "
                      "setpoint. Pass preview= for a moving reference.",
                      file=sys.stderr)
            pr, vr = np_.asarray(xr_seq[0][0:3], float), np_.asarray(xr_seq[0][3:6], float)
            preview = np_.tile(np_.concatenate([pr - p, vr]), (3, 1))
        preview = np_.asarray(preview, float).reshape(3, 6)

        o = np_.concatenate([
            np_.asarray(e, float).reshape(-1),
            preview.reshape(-1),
            np_.clip(self.int_ep, -5.0, 5.0),
            self.du_bar, self.ev_bar, self.om_bar])

        if self.oracle_n == 6:
            w = np_.zeros(6) if d_hat is None else np_.asarray(d_hat, float).reshape(6)
            o = np_.concatenate([o, w if self.oracle_target == "wrench"
                                 else bd_bridge.wrench_to_dmod(w, self.dmod_mode)])
        if o.size != self.obs_w:
            raise ValueError(f"built a {o.size}-wide observation for an actor "
                             f"expecting {self.obs_w}")
        return o

    def _advance_obs(self, e, u, om):
        """The per-step update of (5.3)'s memory, as the training env does it."""
        a, dt = self._ema_a, float(self.X.P.dt_c)
        e = np.asarray(e, float).reshape(-1)
        du = np.asarray(u, float) - np.asarray(self.last_u, float)
        self.int_ep = np.clip(self.int_ep + e[0:3] * dt, -5.0, 5.0)
        self.du_bar = a * self.du_bar + (1 - a) * du
        self.ev_bar = a * self.ev_bar + (1 - a) * e[3:6]
        self.om_bar = a * self.om_bar + (1 - a) * np.asarray(om, float).reshape(3)

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

    def warmup(self):
        """Compile the solve BEFORE the control loop starts.  -> seconds taken.

        JAX traces and compiles on the first call.  Measured on a CPU-only
        jaxlib that is **13.5 s**, against a steady-state 7.4 ms.  Paying it
        inside the first timer callback blocks the executor for those 13.5 s,
        so no ``OffboardControlMode`` is published while it happens -- and PX4
        refuses to enter offboard, or drops straight out of it, if that stream
        stops for more than \SI{0.5}{\second}.  The vehicle would never arm,
        and the log would show a controller that looked healthy.

        So compile against a synthetic hover, here, where taking 13 s costs
        nothing.  The state this leaves behind is discarded: warm-up must not
        seed the rotor observer or (5.3)'s memory with a fictitious step.
        """
        X, np_ = self.X, np
        if self.mode == "pid":
            return 0.0
        t0 = time.perf_counter()
        xr0 = np_.concatenate([np_.zeros(3), np_.zeros(3), [1.0, 0.0, 0.0, 0.0],
                               np_.zeros(3), np_.full(4, float(X.OM_HOVER))])
        xr = np_.tile(xr0, (self.N + 1, 1))
        ur = np_.tile([float(X.U_HOVER), 0.0, 0.0, 0.0], (self.N, 1))
        x13 = np_.concatenate([np_.zeros(3), np_.zeros(3),
                               [1.0, 0.0, 0.0, 0.0], np_.zeros(3)])
        self.step(x13, xr, ur,
                  d_hat=(np_.zeros(6) if self.use_d else None),
                  preview=np_.zeros((3, 6)))
        self.reset()
        self.solve_ms.clear()
        return time.perf_counter() - t0

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

    def step(self, x13, xr_seq, uref_seq, obs=None, d_hat=None, preview=None):
        """x13 = [p, v, q, om] ENU/FLU; xr_seq (N+1,17); uref_seq (N,4).

        ``xr_seq`` comes from :mod:`refgen`, which mirrors the study's
        ``ref_state``.  ``preview`` is the (3, 6) reference lookahead block of
        (5.3); when it is None a position hold is assumed.  ``obs`` overrides
        the built observation entirely, for tests.  Returns
        ``(u_ctbr (4,), info)``.
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
        o_np = None
        if self.mode == "pid":
            u, _ = self.pid(None, jnp.asarray(e), jnp.asarray(xr_seq[0])[None],
                            uref=us[:, 0])
        else:
            #  Only the learned cost map reads the observation.  nmpc1 solves a
            #  fixed quadratic and ignores it, so do not pay to build one.
            if self.mode == "acmpc":
                o_np = (np.asarray(obs, float).reshape(-1) if obs is not None
                        else self._build_obs(e[0], x17, xr_seq, preview, d_hat))
                o = jnp.asarray(o_np)[None]
            else:
                o = jnp.zeros((1, X.OBS_DIM))
            u = self._solver(d is None)(jnp.asarray(e), xs, us, d, o)
        u = np.asarray(u)[0]
        # advance the rotor observer with the command actually issued, through
        # the same model the solver predicted with
        self.Om_hat = np.asarray(
            X.step_c(jnp.asarray(x17)[None], jnp.asarray(u)[None]))[0, 13:17]
        #  Advance (5.3)'s memory with the command actually issued, BEFORE
        #  last_u is overwritten -- du_bar is a mean of u - u_prev.
        self._advance_obs(e[0], u, x17[10:13])
        ms = 1e3 * (time.perf_counter() - t0)
        self.solve_ms.append(ms)
        self.last_u = u
        return u, dict(solve_ms=ms, e=e[0], d_used=None if d is None else np.asarray(d)[0],
                       Om_hat=self.Om_hat.copy(), obs=o_np)
