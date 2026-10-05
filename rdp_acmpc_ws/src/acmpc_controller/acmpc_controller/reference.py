"""Reference sequence and AC-MPC observation, built with the study's functions.

Two things the node must reproduce *exactly* as the policy saw them in training:

* the reference ``(x_ref, u_ref)`` of ``x500_core_jax.ref_state`` -- attitude by
  differential flatness (4.7), ``omega_ref`` by a centred difference, and the
  collective of (4.8) **including the PX4 idle floor** (A2).  The node used to
  compute ``u_ref = sqrt(m |a+g| / T_max)``, the bare root that AUDIT A2 shows is
  5.6 % high at hover (0.769 instead of 0.729), and it dropped ``omega_ref``;
* the 40-D observation of ``x500_core_jax.build_obs`` (5.3) -- error, three
  preview blocks, the clipped position integral and three running means -- plus
  the 6-D wrench channel when the checkpoint was trained with it (variant C).
  The node used to pass no observation, so the learned cost map was evaluated
  on an all-zero input.

``tests/test_ros_wiring.py`` asserts both against the study functions.
"""
from __future__ import annotations

import numpy as np

from . import _study_path                                 # noqa: F401


def _X():
    import x500_core_jax as X
    return X


_REF_CORE = {}


def _ref_core(hold):
    """Jitted (4.7)-(4.8) given the path samples; one compile per ``hold``.

    Built from the study's own ``ref_attitude``/``qmul``/``qconj``.  Un-jitted,
    these few array ops cost ~8 ms per tick in dispatch alone -- 40 % of the
    20 ms period -- so they are traced once, like the solve."""
    if hold not in _REF_CORE:
        import jax
        import jax.numpy as jnp
        X = _X()

        def core(a, a_plus, a_minus):
            if hold:
                a = jnp.zeros_like(a)
            q = X.ref_attitude(a)
            if hold:
                qdot = jnp.zeros_like(q)
            else:
                qdot = (X.ref_attitude(a_plus) - X.ref_attitude(a_minus)) / X.P.dt_c
            om_ref = 2.0 * X.qmul(X.qconj(q), qdot)[..., 1:4]
            r = X.P.Om_min / X.P.Om_max
            root = jnp.sqrt(jnp.clip(X.M_TOT * jnp.linalg.norm(
                a + X.P.g * jnp.asarray(X.E3), axis=-1, keepdims=True) / X.T_MAX, 0.0, 1.0))
            c = jnp.clip((root - r) / (1.0 - r), 0.0, 1.0)
            return q, jnp.concatenate([c, om_ref / jnp.asarray(X.OM_MAX)], -1)
        _REF_CORE[hold] = jax.jit(core)
    return _REF_CORE[hold]


def ref_sequence(pva, t, hold=False):
    """``x500_core_jax.ref_state`` for an arbitrary analytic path.

    pva  : callable t (n,) -> (p, v, a), each (n, 3), ENU
    t    : times (n,)
    hold : position hold -- v = a = 0, so q_ref = identity and u_ref = hover
    Returns ``x_ref (n, 10) = [p, v, q]`` and ``u_ref (n, 4)``.
    """
    X = _X()
    t = np.atleast_1d(np.asarray(t, float))
    p, v, a = (np.asarray(z, float) for z in pva(t))
    if hold:
        v = np.zeros_like(v)
        a_plus = a_minus = a
    else:
        h = 0.5 * X.P.dt_c          # centred difference at +/- dt_c/2 (§5.6)
        a_plus = np.asarray(pva(t + h)[2], float)
        a_minus = np.asarray(pva(t - h)[2], float)
    q, u_ref = (np.asarray(z) for z in _ref_core(bool(hold))(a, a_plus, a_minus))
    return np.concatenate([p, v, q], -1), u_ref


class ObsBuilder:
    """Running state ``prev`` of the study's ``Env`` and the (5.3) observation.

    Call :meth:`update` once per tick with the current error, body rate and the
    command applied since the last tick, *then* :meth:`obs`.  That is the order
    of ``_step_jit`` (which updates ``prev`` with the next-step error) followed
    by ``_obs_jit`` on the next tick.
    """

    def __init__(self, pva, hold=False, u0=None):
        X = _X()
        self.X, self.pva, self.hold = X, pva, bool(hold)
        self.dt = X.P.dt_c
        self.a = float(np.exp(-X.P.dt_c / X.EMA_TAU))
        u0 = np.array([X.U_HOVER, 0.0, 0.0, 0.0]) if u0 is None else np.asarray(u0, float)
        self.prev = dict(int_ep=np.zeros(3), du_bar=np.zeros(4), ev_bar=np.zeros(3),
                         om_bar=np.zeros(3), u=u0.copy())

    def update(self, e, om, du_applied, u_applied):
        a, pr = self.a, self.prev
        e = np.asarray(e, float)
        pr["int_ep"] = np.clip(pr["int_ep"] + e[0:3] * self.dt, -5.0, 5.0)
        pr["du_bar"] = a * pr["du_bar"] + (1 - a) * np.asarray(du_applied, float)
        pr["ev_bar"] = a * pr["ev_bar"] + (1 - a) * e[3:6]
        pr["om_bar"] = a * pr["om_bar"] + (1 - a) * np.asarray(om, float)
        pr["u"] = np.asarray(u_applied, float).copy()

    def obs(self, e, p, t, oracle=None):
        X, pr = self.X, self.prev
        blocks = []
        for h in X.PREVIEW_H:
            t_h = t if self.hold else t + h * X.PREVIEW_STRIDE * X.P.dt_c
            p_h, v_h, _ = self.pva(np.atleast_1d(t_h))
            blocks += [np.asarray(p_h, float)[0] - np.asarray(p, float),
                       np.asarray(v_h, float)[0]]
        o = np.concatenate([np.asarray(e, float), *blocks,
                            np.clip(pr["int_ep"], -5.0, 5.0),
                            pr["du_bar"], pr["ev_bar"], pr["om_bar"]])
        if oracle is not None:
            o = np.concatenate([o, np.asarray(oracle, float)])
        return o
