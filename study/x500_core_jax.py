"""x500_core_jax.py -- core model, optimiser and environment for the Adaptive
AC-MPC study on a Holybro X500.

Implements §2-§5 of AGENTS_SPEC_Adaptive_ACMPC.md.  Pure JAX: this module must
import with no ROS and no PyTorch on the path.  float64 throughout, because the
iLQR backward pass inverts Q_uu at every stage and the learned cost matrices
span ~5 orders of magnitude.

Conventions, fixed once (§3.1):
    world ENU, body FLU, quaternion Hamilton scalar-first, body->world.

Five statements of the specification are corrected here; see docs/CORRECTIONS.md
and study/tests/test_spec_corrections.py.  The ones that touch this file are
C-1/C-2 (the hover linearisation (5.6)) and the notes N-1..N-5.

Every array is batched with a leading axis B.
"""
from __future__ import annotations

import functools
import json
import os
import pickle
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------- #
# §5.1  constants and the parameter block
# --------------------------------------------------------------------------- #
ARTIFACTS = os.environ.get(
    "X500_ARTIFACTS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"),
)


@dataclass(frozen=True)
class P:
    """Immutable physical constants (§2.1) plus the modelling additions.

    Values in this block are read from the PX4 Gazebo SDF description of the
    X500 and are the *only* numbers that may be pasted.  Everything in §2.2 is
    derived from them at import time by :func:`_derive`.
    """

    m_body: float = 2.0
    m_rotor: float = 0.016076923076923075
    n_rotor: int = 4
    K_T: float = 8.54858e-06
    k_m: float = 0.016
    Om_max: float = 1000.0
    tau_up: float = 0.0125
    tau_dn: float = 0.0250
    g: float = 9.8066
    dt_c: float = 0.02
    n_sub: int = 10
    f_gyro: float = 40.0
    rate_max: float = float(np.deg2rad(220.0))
    K_rate: tuple = (14.0, 14.0, 8.0)
    K_i: tuple = (4.0, 4.0, 2.0)
    K_d: tuple = (0.28, 0.28, 0.10)
    I_lim: float = 3.0
    # --- declared modelling additions, NOT from the SDF (§2.1) ---------------
    D: tuple = (0.30, 0.30, 0.35)          # translational drag [N s/m]
    om_max: tuple = (10.0, 10.0, 4.0)      # CTBR rate scaling (3.2) [rad/s]
    J_body: tuple = (0.02166666666666667, 0.02166666666666667, 0.04)
    alpha_feas: float = 0.60               # reference demand cap (§2.2)


#: rotor positions in the body frame (FLU) and spin signs, +1 = CCW from above
R_ROTOR = np.array([[+0.174, -0.174, 0.06],
                    [-0.174, +0.174, 0.06],
                    [+0.174, +0.174, 0.06],
                    [-0.174, -0.174, 0.06]])
SIGMA = np.array([+1.0, +1.0, -1.0, -1.0])

NX, NU, NE, NS = 10, 4, 9, 23          # control state, input, error, plant state
NTAU = NE + NU                          # 13, the cost-map block size (5.2)
E3 = np.array([0.0, 0.0, 1.0])


def _inertia_shift(mass: float, d: np.ndarray) -> np.ndarray:
    """Parallel-axis term  m(||d||^2 I - d d^T)  for a point mass at offset d."""
    return mass * ((d @ d) * np.eye(3) - np.outer(d, d))


def _alloc(arms: np.ndarray, k_m: float, sigma: np.ndarray) -> np.ndarray:
    """Allocation matrix (2.7) from rotor arms measured to the CG."""
    return np.vstack([np.ones(4), arms[:, 1], -arms[:, 0], -sigma * k_m])


def _derive() -> dict:
    """Everything in §2.2-§2.4, computed.  Nothing here may be pasted."""
    m = P.m_body + P.n_rotor * P.m_rotor                                  # (2.1)
    cg_nom = P.m_rotor * R_ROTOR.sum(0) / m                               # (2.2)
    J_nom = np.diag(P.J_body) + sum(                                      # (2.3)
        _inertia_shift(P.m_rotor, r - cg_nom) for r in R_ROTOR)
    f_max = P.K_T * P.Om_max ** 2                                         # (2.4)
    T_max = P.n_rotor * f_max
    u_hover = float(np.sqrt(m * P.g / T_max))                             # (2.5)
    a_lat_max = float(np.sqrt((T_max / m) ** 2 - P.g ** 2))               # (2.6)
    arms = R_ROTOR - cg_nom
    M_nom = _alloc(arms, P.k_m, SIGMA)                                    # (2.7)
    return dict(m=m, cg_nom=cg_nom, J_nom=J_nom, Jinv_nom=np.linalg.inv(J_nom),
                f_max=f_max, T_max=T_max, TW=T_max / (m * P.g), u_hover=u_hover,
                a_lat_max=a_lat_max, M_nom=M_nom, Minv_nom=np.linalg.inv(M_nom),
                Mtau_nom=M_nom[1:], arms_nom=arms)


_D = _derive()
M_TOT, CG_NOM, J_NOM, JINV_NOM = _D["m"], _D["cg_nom"], _D["J_nom"], _D["Jinv_nom"]
F_MAX, T_MAX, TW, U_HOVER = _D["f_max"], _D["T_max"], _D["TW"], _D["u_hover"]
A_LAT_MAX = _D["a_lat_max"]
M_NOM, MINV_NOM, MTAU_NOM, ARMS_NOM = _D["M_nom"], _D["Minv_nom"], _D["Mtau_nom"], _D["arms_nom"]

# The J off-diagonals must vanish by symmetry (§2.2); assert it at import so a
# bad rotor table cannot pass silently.
assert np.abs(J_NOM - np.diag(np.diag(J_NOM))).max() < 1e-15, "J off-diagonals nonzero"
assert abs(float(np.linalg.cond(M_NOM)) - 62.5) < 1e-6, "cond(M) != 62.5"

OM_MAX = np.asarray(P.om_max)
DAZ_DC_HOVER = 2.0 * T_MAX * U_HOVER / M_TOT       # (2.11) == 2 g / u_hover


def lateral_accel_budget() -> float:
    """(2.6) the horizontal acceleration left once weight is carried [m/s^2]."""
    return A_LAT_MAX


def _b(x, B):
    """Broadcast a constant to a leading batch axis of size B."""
    return jnp.broadcast_to(jnp.asarray(x, dtype=jnp.float64), (B,) + jnp.shape(x))


def make_par(B: int = 1, m_scale=1.0, D_scale=1.0, tau_scale=1.0, Tmax_scale=1.0,
             Kw_scale=1.0, J_scale=1.0, payload_m=0.0, payload_r=None,
             wrench_body=None) -> dict:
    """Batched plant parameters; implements (3.10)-(3.11) and rebuilds the arms.

    Every scale may be a scalar or a length-B array.  Two mixer entries are
    carried, per the boxed warning of §3.3:

    ``J``/``Jinv``/``Mtau_w2`` are the **true** geometry and drive (3.6)-(3.7);
    ``J_ctrl``/``Minv_ctrl`` are the **nominal** geometry and drive the inner
    loop (3.4)-(3.5).  A real PX4 allocator has fixed parameters and does not
    know a payload was attached -- if the mixer knows, an off-centre payload is
    allocated away before it is felt and the whole adaptive half goes trivial.

    ``Tmax_scale`` (N-4) perturbs only the rotors: ``KT`` is scaled, while the
    allocator and the control model keep ``T_max_ctrl`` = nominal.
    """
    f = lambda v: jnp.broadcast_to(jnp.asarray(v, dtype=jnp.float64).reshape(-1), (B,))
    m_s, D_s, tau_s = f(m_scale), f(D_scale), f(tau_scale)
    T_s, Kw_s, J_s = f(Tmax_scale), f(Kw_scale), f(J_scale)
    p_m = f(payload_m)
    p_r = (jnp.zeros((B, 3)) if payload_r is None
           else jnp.broadcast_to(jnp.asarray(payload_r, dtype=jnp.float64), (B, 3)))

    m_body = P.m_body * m_s                                    # airframe only
    m_rotors = P.n_rotor * P.m_rotor * m_s
    m_air = m_body + m_rotors                                  # scaled airframe
    m_t = m_air + p_m                                          # (3.10) total
    cg0 = jnp.broadcast_to(jnp.asarray(CG_NOM), (B, 3))
    cg = (m_air[:, None] * cg0 + p_m[:, None] * p_r) / m_t[:, None]

    d_b, d_p = cg0 - cg, p_r - cg
    def _shift(mass, d):
        return mass[:, None, None] * (
            jnp.sum(d * d, -1)[:, None, None] * jnp.eye(3)
            - d[:, :, None] * d[:, None, :])
    J = (J_s[:, None, None] * m_s[:, None, None] * jnp.asarray(J_NOM)      # (3.11)
         + _shift(m_air, d_b) + _shift(p_m, d_p))
    J_ctrl = jnp.broadcast_to(jnp.asarray(J_NOM), (B, 3, 3))

    arms = jnp.asarray(R_ROTOR)[None] - cg[:, None, :]          # rebuilt arms
    Mtau = jnp.stack([arms[..., 1], -arms[..., 0],
                      -jnp.broadcast_to(jnp.asarray(SIGMA) * P.k_m, (B, 4))], axis=1)
    KT = P.K_T * T_s                                            # N-4
    return dict(
        m=m_t[:, None], J=J, Jinv=jnp.linalg.inv(J), J_ctrl=J_ctrl,
        Mtau_w2=Mtau * KT[:, None, None],
        Minv_ctrl=jnp.broadcast_to(jnp.asarray(MINV_NOM), (B, 4, 4)),
        KT=KT[:, None], T_max=(4.0 * KT * P.Om_max ** 2)[:, None],
        T_max_ctrl=jnp.full((B, 1), T_MAX), m_ctrl=jnp.full((B, 1), M_TOT),
        tau_up=(P.tau_up * tau_s)[:, None], tau_dn=(P.tau_dn * tau_s)[:, None],
        D=jnp.asarray(P.D) * D_s[:, None],
        K_rate=jnp.asarray(P.K_rate) * Kw_s[:, None],
        K_i=_b(np.asarray(P.K_i), B), K_d=_b(np.asarray(P.K_d), B),
        cg=cg, arms=arms,
        wrench_body=(jnp.zeros((B, 6)) if wrench_body is None
                     else jnp.broadcast_to(jnp.asarray(wrench_body, dtype=jnp.float64), (B, 6))),
    )


def par_set_wrench(par: dict, w: jnp.ndarray) -> dict:
    """Return ``par`` with the body-frame external wrench replaced."""
    return {**par, "wrench_body": w}


# --------------------------------------------------------------------------- #
# §5.2  quaternion utilities
# --------------------------------------------------------------------------- #
def qmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product (3.1), (B,4) x (B,4) -> (B,4)."""
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return jnp.concatenate(
        [aw * bw - jnp.sum(av * bv, -1, keepdims=True),
         aw * bv + bw * av + jnp.cross(av, bv)], axis=-1)


def qconj(q: jnp.ndarray) -> jnp.ndarray:
    return q * jnp.asarray([1.0, -1.0, -1.0, -1.0])


def qnorm(q: jnp.ndarray) -> jnp.ndarray:
    """Renormalise and sign-fix so that q_w >= 0 (the shorter rotation)."""
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    return q * jnp.where(q[..., :1] < 0.0, -1.0, 1.0)


def qrotmat(q: jnp.ndarray) -> jnp.ndarray:
    """(B,4) -> (B,3,3) body->world rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack([
        jnp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        jnp.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        jnp.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def qzaxis(q: jnp.ndarray) -> jnp.ndarray:
    """Third column of qrotmat(q): the body-up axis expressed in the world."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)


def qexp_tilt(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    """(4.7): the quaternion of a rotation ``angle`` about the unit ``axis``."""
    n = jnp.linalg.norm(axis, axis=-1, keepdims=True)
    u = jnp.where(n > 1e-12, axis / jnp.where(n > 0, n, 1.0), jnp.zeros_like(axis))
    h = 0.5 * angle[..., None]
    return jnp.concatenate([jnp.cos(h), u * jnp.sin(h)], -1)


# --------------------------------------------------------------------------- #
# N-2  the clamp and its gradient convention (§5.8)
# --------------------------------------------------------------------------- #
@jax.custom_jvp
def _clamp(x, lo, hi):
    """clip with an explicit, documented derivative at the box boundary.

    d/dx = 1 for ``lo <= x <= hi`` -- pass-through **on** the boundary -- and 0
    strictly outside.  ``jnp.clip`` and ``torch.clamp`` disagree here, and the
    iLQR line search parks the collective exactly on the edge, so this is not a
    corner case: choosing 0 zeroes the collective column of B precisely when the
    input is on the box, making Q_uu singular in that direction.  T-7 therefore
    compares against a *one-sided* difference into the feasible interior at the
    boundary; a central difference across the kink would return 1/2.
    """
    return jnp.clip(x, lo, hi)


@_clamp.defjvp
def _clamp_jvp(primals, tangents):
    x, lo, hi = primals
    dx = tangents[0]
    inside = (x >= lo) & (x <= hi)
    return jnp.clip(x, lo, hi), jnp.where(inside, dx, jnp.zeros_like(dx))


def thrust_of(c, T_max):
    """(2.10) velocity-type motor model: thrust is quadratic in the command."""
    return T_max * _clamp(c, 0.0, 1.0) ** 2


# --------------------------------------------------------------------------- #
# §5.3  control model -- 10 states, what the optimiser predicts with
# --------------------------------------------------------------------------- #
def _ctrl_consts(par):
    if par is None:
        return M_TOT, T_MAX
    return par["m_ctrl"], par["T_max_ctrl"]


def fc(x, u, d=None, par=None):
    """(3.3) control-model derivative.  x (B,10), u (B,4), d (B,6) -> (B,10).

    ``d = [a_res, om_res]`` in [m/s^2, rad/s] is the optional model correction of
    §4.3(a).  It is **not** a wrench: feed :func:`wrench_to_dmod` output here,
    never :func:`external_wrench` output (§4.4).
    """
    m, T_max = _ctrl_consts(par)
    v, q = x[..., 3:6], x[..., 6:10]
    om = jnp.asarray(OM_MAX) * u[..., 1:4]                                # (3.2)
    a_res = jnp.zeros_like(v) if d is None else d[..., 0:3]
    om_res = jnp.zeros_like(om) if d is None else d[..., 3:6]
    acc = thrust_of(u[..., 0:1], T_max) / m * qzaxis(q) - P.g * jnp.asarray(E3) + a_res
    qdot = 0.5 * qmul(q, jnp.concatenate(
        [jnp.zeros_like(om[..., :1]), om + om_res], -1))
    return jnp.concatenate([v, acc, qdot], -1)


def step_c(x, u, d=None, par=None, dt=None):
    """One classical RK4 step of the control model at dt_c, renormalising q."""
    dt = P.dt_c if dt is None else dt
    k1 = fc(x, u, d, par)
    k2 = fc(x + 0.5 * dt * k1, u, d, par)
    k3 = fc(x + 0.5 * dt * k2, u, d, par)
    k4 = fc(x + dt * k3, u, d, par)
    xn = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return jnp.concatenate([xn[..., :6], qnorm(xn[..., 6:10])], -1)


# --------------------------------------------------------------------------- #
# §5.4  plant -- 23 states, the truth
# --------------------------------------------------------------------------- #
# s = [p(3) | q(4) | v(3) | om(3) | Omega(4) | I_om(3) | om_f(3)]
SP, SQ, SV, SW, SO, SI, SF = (slice(0, 3), slice(3, 7), slice(7, 10), slice(10, 13),
                              slice(13, 17), slice(17, 20), slice(20, 23))


def rate_cmd(u_ctbr):
    """Commanded body rate: (3.2) then the 220 deg/s autopilot clip (§3.3)."""
    return _clamp(jnp.asarray(OM_MAX) * u_ctbr[..., 1:4], -P.rate_max, P.rate_max)


def inner_loop(s, u_ctbr, par):
    """(3.4)-(3.5).  Returns (Omega_cmd (B,4), pwm (B,4), tau_cmd (B,3)).

    Uses ``par['Minv_ctrl']`` and ``par['J_ctrl']`` -- the **nominal** geometry --
    and the nominal T_max.  Never the true ones (§3.3, N-4).
    """
    q, om, I_om, om_f = s[..., SQ], s[..., SW], s[..., SI], s[..., SF]
    om_c = rate_cmd(u_ctbr)
    Jc = par["J_ctrl"]
    ang = (par["K_rate"] * (om_c - om_f) + par["K_i"] * I_om - par["K_d"] * om_f)
    tau_cmd = (jnp.einsum("bij,bj->bi", Jc, ang)
               + jnp.cross(om, jnp.einsum("bij,bj->bi", Jc, om)))
    T_c = thrust_of(u_ctbr[..., 0:1], par["T_max_ctrl"])
    f_cmd = _clamp(jnp.einsum("bij,bj->bi", par["Minv_ctrl"],
                              jnp.concatenate([T_c, tau_cmd], -1)), 0.0, F_MAX)
    Om_cmd = jnp.sqrt(f_cmd / P.K_T)          # allocator uses the NOMINAL K_T
    Om_cmd = _clamp(Om_cmd, 0.0, P.Om_max)
    return Om_cmd, Om_cmd / P.Om_max, tau_cmd


def fp(s, u_ctbr, par, v_wind=None):
    """(3.4)-(3.8) plant derivative.  (B,23) -> (B,23).

    Rigid body and rotors use the **true** geometry ``par['J']``,
    ``par['Jinv']``, ``par['Mtau_w2']``; the inner loop uses the nominal one.
    """
    q, v, om = s[..., SQ], s[..., SV], s[..., SW]
    Om, om_f = s[..., SO], s[..., SF]
    B = s.shape[0]
    v_w = jnp.zeros((B, 3)) if v_wind is None else v_wind
    Om_cmd, _, _ = inner_loop(s, u_ctbr, par)

    R = qrotmat(q)
    w2 = Om ** 2
    thrust = jnp.sum(par["KT"] * w2, -1, keepdims=True)
    F_b, tau_b = par["wrench_body"][..., 0:3], par["wrench_body"][..., 3:6]
    acc = (thrust / par["m"] * qzaxis(q) - P.g * jnp.asarray(E3)            # (3.6)
           - par["D"] * (v - v_w) / par["m"]
           + jnp.einsum("bij,bj->bi", R, F_b) / par["m"])
    tau = (jnp.einsum("bij,bj->bi", par["Mtau_w2"], w2) + tau_b             # (3.7)
           - jnp.cross(om, jnp.einsum("bij,bj->bi", par["J"], om)))
    omdot = jnp.einsum("bij,bj->bi", par["Jinv"], tau)
    qdot = 0.5 * qmul(q, jnp.concatenate([jnp.zeros((B, 1)), om], -1))
    tau_m = jnp.where(Om_cmd >= Om, par["tau_up"], par["tau_dn"])           # (3.8)
    Omdot = (Om_cmd - Om) / tau_m
    om_fdot = 2 * jnp.pi * P.f_gyro * (om - om_f)
    I_dot = rate_cmd(u_ctbr) - om_f
    return jnp.concatenate([v, qdot, acc, omdot, Omdot, I_dot, om_fdot], -1)


def _plant_project(s):
    """Renormalise q, keep rotor speeds physical, clamp the integrator (N-5)."""
    return jnp.concatenate([
        s[..., SP], qnorm(s[..., SQ]), s[..., SV], s[..., SW],
        jnp.clip(s[..., SO], 0.0, P.Om_max),
        jnp.clip(s[..., SI], -P.I_lim, P.I_lim), s[..., SF]], -1)


def step_p(s, u_ctbr, par, v_wind=None, n_sub=None):
    """``n_sub`` RK4 sub-steps of dt_c/n_sub (2 ms by default)."""
    n_sub = P.n_sub if n_sub is None else n_sub
    h = P.dt_c / n_sub

    def body(_, st):
        k1 = fp(st, u_ctbr, par, v_wind)
        k2 = fp(st + 0.5 * h * k1, u_ctbr, par, v_wind)
        k3 = fp(st + 0.5 * h * k2, u_ctbr, par, v_wind)
        k4 = fp(st + h * k3, u_ctbr, par, v_wind)
        return _plant_project(st + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    return jax.lax.fori_loop(0, n_sub, body, s)


def plant_to_ctrl(s):
    """The 10-state view of a 23-state plant state: [p, v, q]."""
    return jnp.concatenate([s[..., SP], s[..., SV], s[..., SQ]], -1)


def hover_state(B=1, p=(0.0, 0.0, 1.5), par=None):
    """A plant state trimmed for hover: rotors spun up, integrator at rest.

    Built so that T-8 can be evaluated where it is meaningful -- the residual is
    only identically zero once the rotors have reached their commanded speed.
    """
    par = make_par(B) if par is None else par
    p = jnp.broadcast_to(jnp.asarray(p, dtype=jnp.float64), (B, 3))
    q = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0]), (B, 4))
    f_h = par["m"] * P.g / 4.0
    Om = jnp.broadcast_to(jnp.sqrt(f_h / par["KT"]), (B, 4))
    z = jnp.zeros((B, 3))
    return jnp.concatenate([p, q, z, z, Om, z, z], -1)


def hover_u(B=1, par=None):
    """The CTBR command that trims :func:`hover_state`."""
    par = make_par(B) if par is None else par
    c = jnp.sqrt(par["m"] * P.g / par["T_max_ctrl"])
    return jnp.concatenate([c, jnp.zeros((B, 3))], -1)


# --------------------------------------------------------------------------- #
# §5.5  error coordinates
# --------------------------------------------------------------------------- #
def err(x, xr):
    """(4.1) reduced error, (B,10),(B,10) -> (B,9).

    The sign of q_e picks the shorter rotation, so ||e[6:9]|| = 2 sin(theta/2).
    """
    qe = qmul(qconj(xr[..., 6:10]), x[..., 6:10])
    sgn = jnp.where(qe[..., :1] >= 0.0, 1.0, -1.0)
    return jnp.concatenate([x[..., 0:3] - xr[..., 0:3],
                            x[..., 3:6] - xr[..., 3:6], 2.0 * sgn * qe[..., 1:4]], -1)


def e_to_state(e, xr):
    """(4.2) the exact inverse of :func:`err`, (B,9),(B,10) -> (B,10).

    N-1: implemented in the algebraically identical closed form
    ``q_e = [sqrt(1 - ||d||^2/4), d/2]``.  Substituting n = 2 arcsin(||d||/2)
    into (4.2) gives cos(n/2) = sqrt(1 - ||d||^2/4) and d_hat sin(n/2) = d/2, so
    the two agree to 1e-14 -- but the spec's form divides by ||d||, whose
    gradient is NaN at zero error, which is exactly the point the iLQR
    linearises about at convergence.  Do **not** use the shortcut n = ||d||: it
    is wrong by -2.70 deg at 60 deg and -20.8 deg at 120 deg.
    """
    dlt = e[..., 6:9]
    n2 = jnp.sum(dlt * dlt, -1, keepdims=True)
    qw = jnp.sqrt(jnp.clip(1.0 - n2 / 4.0, 0.0, 1.0))
    qe = jnp.concatenate([qw, dlt / 2.0], -1)
    return jnp.concatenate([e[..., 0:3] + xr[..., 0:3],
                            e[..., 3:6] + xr[..., 3:6], qmul(xr[..., 6:10], qe)], -1)


# --------------------------------------------------------------------------- #
# §5.6  references
# --------------------------------------------------------------------------- #
PATHS_ALL = ("hover", "step", "circle", "fig8", "helix", "square")
PATH_IDX = {k: i for i, k in enumerate(PATHS_ALL)}
HELIX_CLIMB = 0.25          # climb rate as a fraction of speed (§4.2)


def _superellipse(th):
    """(4.4)-(4.5): r, r', r'' for r = (cos^8 th + sin^8 th)^{-1/8}.

    The sign of the *second* term of u' is ``+``.  Writing it as ``-`` changes
    kappa_a silently; KAPPA_A below is recomputed by dense sampling at import
    and asserted against the §4.2 table so that cannot happen (C-5).
    """
    c, s = jnp.cos(th), jnp.sin(th)
    f = c ** 8 + s ** 8                                   # in [2^-3, 1], never 0
    u = s ** 7 * c - c ** 7 * s
    up = 7 * s ** 6 * c ** 2 - s ** 8 + 7 * c ** 6 * s ** 2 - c ** 8
    r = f ** (-1.0 / 8.0)
    rp = -f ** (-9.0 / 8.0) * u
    rpp = 9 * f ** (-17.0 / 8.0) * u ** 2 - f ** (-9.0 / 8.0) * up
    return r, rp, rpp


def _path_shape(kind_i, th):
    """Unit-R, unit-omega shape and its first two theta-derivatives, (B,3) each."""
    c, s = jnp.cos(th), jnp.sin(th)
    z = jnp.zeros_like(th)
    if kind_i in (0, 1):                                   # hover / step
        return (jnp.stack([z, z, z], -1),) * 3
    if kind_i in (2, 4):                                   # circle / helix
        return (jnp.stack([c, s, z], -1), jnp.stack([-s, c, z], -1),
                jnp.stack([-c, -s, z], -1))
    if kind_i == 3:                                        # fig8, Gerono
        return (jnp.stack([s, s * c, z], -1),
                jnp.stack([c, jnp.cos(2 * th), z], -1),            # (4.3)
                jnp.stack([-s, -2 * jnp.sin(2 * th), z], -1))
    if kind_i == 5:                                        # superellipse
        r, rp, rpp = _superellipse(th)
        return (jnp.stack([r * c, r * s, z], -1),
                jnp.stack([rp * c - r * s, rp * s + r * c, z], -1),
                jnp.stack([rpp * c - 2 * rp * s - r * c,
                           rpp * s + 2 * rp * c - r * s, z], -1))
    raise ValueError(kind_i)


def _kappas():
    """Peak |shape'| and |shape''| at R = omega = 1, by dense sampling."""
    th = jnp.linspace(0.0, 2 * jnp.pi, 200_001)
    kv, ka = [], []
    for i in range(len(PATHS_ALL)):
        if i == 4:                                         # helix, see the dagger
            kv.append(float(np.sqrt(1.0 + HELIX_CLIMB ** 2))); ka.append(1.0); continue
        _, d1, d2 = _path_shape(i, th)
        kv.append(float(jnp.linalg.norm(d1, axis=-1).max()))
        ka.append(float(jnp.linalg.norm(d2, axis=-1).max()))
    return tuple(kv), tuple(ka)


KAPPA_V, KAPPA_A = _kappas()
_KV_SPEC = (0.0, 0.0, 1.0, 1.4142136, 1.0307764, 1.4160340)
_KA_SPEC = (0.0, 0.0, 1.0, 2.1250000, 1.0, 9.0778770)
for _i, _k in enumerate(PATHS_ALL):                    # §4.2 table, verified
    assert abs(KAPPA_V[_i] - _KV_SPEC[_i]) < 1e-6, (_k, KAPPA_V[_i])
    assert abs(KAPPA_A[_i] - _KA_SPEC[_i]) < 1e-6, (_k, KAPPA_A[_i])


def path_omega(kind, R, spd):
    """(4.6) feasibility-capped path rate [rad/s].

    Without the cap the superellipse at R=0.5, spd=1.5 demands 40.85 m/s^2
    against 13.35 available; every controller then fails identically and the
    benchmark measures the reference rather than the controller.
    """
    i = PATH_IDX[kind] if isinstance(kind, str) else kind
    ka = jnp.asarray(KAPPA_A)[i] if not isinstance(i, int) else KAPPA_A[i]
    R = jnp.asarray(R, dtype=jnp.float64)
    w_spd = spd / jnp.maximum(R, 0.3)
    ka = jnp.asarray(ka, dtype=jnp.float64)
    w_feas = jnp.where(ka > 0.0,
                       jnp.sqrt(P.alpha_feas * A_LAT_MAX / jnp.maximum(R * ka, 1e-12)),
                       jnp.inf)
    w = jnp.minimum(w_spd, w_feas)
    moving = jnp.asarray(jnp.where(jnp.asarray(i) >= 2, 1.0, 0.0))
    return w * moving


def path_demand(kind, R, omega):
    """Peak (|v_ref|, |a_ref|) for a path, for reporting (§8.1 step 2)."""
    i = PATH_IDX[kind] if isinstance(kind, str) else kind
    if i == 4:
        spd = R * omega
        return float(np.hypot(spd, HELIX_CLIMB * spd)), float(R * omega ** 2 * KAPPA_A[i])
    return float(R * omega * KAPPA_V[i]), float(R * omega ** 2 * KAPPA_A[i])


def _ref_pva(ep, t):
    """Reference position/velocity/acceleration, (B,3) each.  Exact analytic."""
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    th = ep["omega"] * t + ep["phi0"]
    R, w = ep["R"][:, None], ep["omega"][:, None]
    ps, vs, as_ = [], [], []
    for i in range(len(PATHS_ALL)):
        d0, d1, d2 = _path_shape(i, th)
        p, v, a = R * d0, R * w * d1, R * w * w * d2
        if i == 0:
            p = jnp.zeros_like(p)
        elif i == 1:
            p = ep["delta"]
            v = a = jnp.zeros_like(p)
        elif i == 4:                                          # helix climb term
            climb = HELIX_CLIMB * ep["spd"][:, None] * t[:, None]
            p = p + climb * jnp.asarray(E3)
            v = v + HELIX_CLIMB * ep["spd"][:, None] * jnp.asarray(E3)
        ps.append(p); vs.append(v); as_.append(a)
    sel = lambda L: jnp.take_along_axis(jnp.stack(L, 1), ep["kind"][:, None, None], 1)[:, 0]
    return ep["c"] + sel(ps), sel(vs), sel(as_)


def ref_attitude(a_ref):
    """(4.7) differential-flatness reference attitude, yaw held at zero."""
    zb = a_ref + P.g * jnp.asarray(E3)
    zb = zb / jnp.linalg.norm(zb, axis=-1, keepdims=True)
    ax = jnp.cross(jnp.broadcast_to(jnp.asarray(E3), zb.shape), zb)
    ang = jnp.arccos(jnp.clip(zb[..., 2], -1.0, 1.0))
    return qnorm(qexp_tilt(ax, ang))


def ref_state(ep, t):
    """(B,10) reference state, (B,4) u_ref (4.8), (B,3) a_ref.

    omega_ref comes from a centred difference of q_ref at +/- dt_c/2 (§5.6).
    The collective in (4.8) carries a **square root** -- it is exact, because
    thrust is quadratic in the command (2.10).
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    p, v, a = _ref_pva(ep, t)
    q = ref_attitude(a)
    h = 0.5 * P.dt_c
    qp = ref_attitude(_ref_pva(ep, t + h)[2])
    qm = ref_attitude(_ref_pva(ep, t - h)[2])
    qdot = (qp - qm) / (2 * h)
    om_ref = 2.0 * qmul(qconj(q), qdot)[..., 1:4]              # body rates
    c = jnp.sqrt(jnp.clip(
        M_TOT * jnp.linalg.norm(a + P.g * jnp.asarray(E3), axis=-1, keepdims=True)
        / T_MAX, 0.0, 1.0))
    u_ref = jnp.concatenate([c, om_ref / jnp.asarray(OM_MAX)], -1)
    return jnp.concatenate([p, v, q], -1), u_ref, a


# --------------------------------------------------------------------------- #
# §5.7  disturbance accessors -- (a) model residual vs (b) physical wrench
# --------------------------------------------------------------------------- #
def true_disturbance(s, u, par, v_wind=None, sdot=None):
    """(4.9) model residual d = [a_plant - a_model, om_plant - om_cmd].

    Units **[m/s^2, rad/s]**.  This is object (a) of §4.3 and is what may be fed
    to :func:`fc`.  It is *not* a wrench: see :func:`external_wrench`.
    """
    sdot = fp(s, u, par, v_wind) if sdot is None else sdot
    x = plant_to_ctrl(s)
    a_model = (thrust_of(u[..., 0:1], par["T_max_ctrl"]) / par["m_ctrl"]
               * qzaxis(x[..., 6:10]) - P.g * jnp.asarray(E3))
    return jnp.concatenate([sdot[..., SV] - a_model, s[..., SW] - rate_cmd(u)], -1)


def external_wrench(s, u, par, sdot=None, v_wind=None):
    """(4.10)-(4.11) the physical wrench a force-torque sensor would read.

    Units **[N world, N m body]**.  This is object (b) of §4.3 -- the RDP target
    and the observation channel.  Feeding it to :func:`fc` is the error §4.4
    warns about: it is a factor m too large in force and dimensionally unrelated
    in moment.

    Both blocks are evaluated against the **nominal** mass, inertia and
    allocation, because the residual is by definition what the nominal model
    fails to explain.  Using the true ones makes a payload invisible: at hover
    the extra weight is then exactly cancelled by the extra thrust.
    """
    sdot = fp(s, u, par, v_wind) if sdot is None else sdot
    q, om = s[..., SQ], s[..., SW]
    m_n, J_n = par["m_ctrl"], par["J_ctrl"]
    T = thrust_of(u[..., 0:1], par["T_max_ctrl"])
    F = m_n * sdot[..., SV] - (T * qzaxis(q) - m_n * P.g * jnp.asarray(E3))   # (4.10)
    tau = (jnp.einsum("bij,bj->bi", J_n, sdot[..., SW])                       # (4.11)
           + jnp.cross(om, jnp.einsum("bij,bj->bi", J_n, om))
           - jnp.einsum("ij,bj->bi", jnp.asarray(MTAU_NOM) * P.K_T, s[..., SO] ** 2))
    return jnp.concatenate([F, tau], -1)


def wrench_to_dmod(w, mode: str = "first_order"):
    """(4.12)+(4.13) convert a wrench [N, N m] to a model residual [m/s^2, rad/s].

    Force converts exactly, a_res = F/m.  The moment block has **no exact
    image** -- the 10-state model has no torque input -- so (4.13) uses the
    steady state of the rate loop, om_res ~ diag(K_r)^-1 J^-1 tau.  That is a
    *model*, valid while the horizon is short against the integral time
    constant (1/K_r = 71 ms vs 1/K_i = 250 ms, horizons 20-400 ms).  T-11
    validates it numerically; ``mode='none'`` zeroes the block so both can be
    reported.
    """
    a_res = w[..., 0:3] / M_TOT
    if mode == "none":
        om_res = jnp.zeros_like(a_res)
    elif mode == "first_order":
        om_res = jnp.einsum("ij,bj->bi", jnp.linalg.inv(
            np.diag(P.K_rate) @ J_NOM), w[..., 3:6])
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return jnp.concatenate([a_res, om_res], -1)


# --------------------------------------------------------------------------- #
# §5.8  the differentiable MPC layer
# --------------------------------------------------------------------------- #
U_LO = np.array([0.0, -1.0, -1.0, -1.0])
U_HI = np.array([1.0, 1.0, 1.0, 1.0])
ALPHAS = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
MU_MIN, MU_MAX = 1e-8, 1e10


def uref_from_traj(xr_seq):
    """N-3 fallback: reconstruct u_ref from consecutive reference states.

    (4.7) fixes only the *direction* of a_ref + g e3, so u_ref cannot be read
    off a single xr.  a_ref is recovered to first order from the reference
    velocities; :meth:`Env.ref_useq` supplies the exact analytic value and every
    controller that binds an environment uses that instead.
    """
    v = xr_seq[..., 3:6]
    a = (v[:, 1:] - v[:, :-1]) / P.dt_c
    zb = a + P.g * jnp.asarray(E3)
    c = jnp.sqrt(jnp.clip(M_TOT * jnp.linalg.norm(zb, axis=-1, keepdims=True) / T_MAX,
                          0.0, 1.0))
    return jnp.concatenate([c, jnp.zeros(c.shape[:-1] + (3,))], -1)


def edyn(e, du, xr, xr_next, d=None, par=None, uref=None):
    """(5.1) one step of the error dynamics against the **moving** reference.

    Freezing the reference is a bug, not a simplification: at 1.5 m/s over a
    20-step horizon the target moves 0.6 m, so a frozen-reference solver plans
    to come to rest on the current waypoint and error *grows* with N (T-6).
    """
    if uref is None:
        uref = jnp.zeros_like(du).at[..., 0].set(U_HOVER)
    du = _clamp(du, jnp.asarray(U_LO) - uref, jnp.asarray(U_HI) - uref)
    x = e_to_state(e, xr)
    return err(step_c(x, uref + du, d, par), xr_next)


def rollout_err(e0, du_seq, xr_seq, uref_seq, d=None, par=None):
    """Forward rollout of (5.1).  du_seq (B,N,4) -> e_seq (B,N+1,9)."""
    def body(e, k):
        en = edyn(e, du_seq[:, k], xr_seq[:, k], xr_seq[:, k + 1], d, par, uref_seq[:, k])
        return en, en
    N = du_seq.shape[1]
    eN, es = jax.lax.scan(body, e0, jnp.arange(N))
    return jnp.concatenate([e0[:, None], jnp.swapaxes(es, 0, 1)], 1)


def traj_cost(e_seq, du_seq, S, c, P_term):
    """(5.2) objective value, (B,)."""
    tau = jnp.concatenate([e_seq[:, :-1], du_seq], -1)                 # (B,N,13)
    quad = 0.5 * jnp.einsum("bni,bnij,bnj->b", tau, S, tau)
    lin = jnp.einsum("bni,bni->b", c, tau)
    eN = e_seq[:, -1]
    return quad + lin + 0.5 * jnp.einsum("bi,bij,bj->b", eN, P_term, eN)


def _edyn_row(e, du, xr, xrn, uref, d, par):
    """Single-vehicle (5.1): promotes to a batch of one so :func:`edyn` applies."""
    par1 = None if par is None else jax.tree_util.tree_map(lambda z: z[None], par)
    d1 = None if d is None else d[None]
    return edyn(e[None], du[None], xr[None], xrn[None], d1, par1, uref[None])[0]


def lin_traj(e0, du_seq, xr_seq, d=None, par=None, uref_seq=None, e_seq=None):
    """Stage Jacobians of (5.1): A (B,N,9,9), B (B,N,9,4).

    Taken with ``vmap`` over the batch, not by differentiating the batched map:
    the latter builds a (B,9,B,4) cross-Jacobian that is B times too large in
    both memory and work, and is all zeros off the diagonal.  The stage loop is
    a ``scan``, so the compiled program does not grow with N.
    """
    if uref_seq is None:
        uref_seq = uref_from_traj(xr_seq)
    if e_seq is None:
        e_seq = rollout_err(e0, du_seq, xr_seq, uref_seq, d, par)
    ax_d = None if d is None else 0
    ax_p = None if par is None else 0
    jA = jax.vmap(jax.jacfwd(_edyn_row, 0), in_axes=(0, 0, 0, 0, 0, ax_d, ax_p))
    jB = jax.vmap(jax.jacfwd(_edyn_row, 1), in_axes=(0, 0, 0, 0, 0, ax_d, ax_p))

    def body(_, xs):
        ek, duk, xrk, xrn, urk = xs
        return None, (jA(ek, duk, xrk, xrn, urk, d, par),
                      jB(ek, duk, xrk, xrn, urk, d, par))

    tr = lambda z: jnp.swapaxes(z, 0, 1)
    _, (As, Bs) = jax.lax.scan(
        body, None, (tr(e_seq[:, :-1]), tr(du_seq), tr(xr_seq[:, :-1]),
                     tr(xr_seq[:, 1:]), tr(uref_seq)))
    return tr(As), tr(Bs)


def _backward(A, Bm, S, c, P_term, e_seq, du_seq, mu):
    """iLQR backward pass (Li-Todorov 2004; Tassa et al. 2012 regularisation)."""
    N = du_seq.shape[1]
    Sxx, Sxu, Suu = S[..., :NE, :NE], S[..., :NE, NE:], S[..., NE:, NE:]
    Sux = jnp.swapaxes(Sxu, -1, -2)
    cx, cu = c[..., :NE], c[..., NE:]
    lx = (jnp.einsum("bnij,bnj->bni", Sxx, e_seq[:, :-1])
          + jnp.einsum("bnij,bnj->bni", Sxu, du_seq) + cx)
    lu = (jnp.einsum("bnij,bnj->bni", Sux, e_seq[:, :-1])
          + jnp.einsum("bnij,bnj->bni", Suu, du_seq) + cu)
    V = P_term
    v = jnp.einsum("bij,bj->bi", P_term, e_seq[:, -1])
    eye_u = jnp.eye(NU)

    def body(carry, k):
        V, v = carry
        Ak, Bk = A[:, k], Bm[:, k]
        Qx = lx[:, k] + jnp.einsum("bji,bj->bi", Ak, v)
        Qu = lu[:, k] + jnp.einsum("bji,bj->bi", Bk, v)
        VA = jnp.einsum("bij,bjk->bik", V, Ak)
        Qxx = Sxx[:, k] + jnp.einsum("bji,bjk->bik", Ak, VA)
        Quu = (Suu[:, k] + jnp.einsum("bji,bjk,bkl->bil", Bk, V, Bk)
               + mu[:, None, None] * eye_u)
        Qux = Sux[:, k] + jnp.einsum("bji,bjk->bik", Bk, VA)
        sol = jnp.linalg.solve(Quu, jnp.concatenate([Qux, Qu[..., None]], -1))
        K, kff = -sol[..., :NE], -sol[..., NE]
        KtQuu = jnp.einsum("bji,bjk->bik", K, Quu)
        Vn = (Qxx + jnp.einsum("bij,bjk->bik", KtQuu, K)
              + jnp.einsum("bji,bjk->bik", K, Qux)
              + jnp.einsum("bji,bjk->bik", Qux, K))
        Vn = 0.5 * (Vn + jnp.swapaxes(Vn, -1, -2))
        vn = (Qx + jnp.einsum("bij,bj->bi", KtQuu, kff)
              + jnp.einsum("bji,bj->bi", K, Qu)
              + jnp.einsum("bji,bj->bi", Qux, kff))
        dV = (jnp.einsum("bi,bi->b", kff, Qu)
              + 0.5 * jnp.einsum("bi,bij,bj->b", kff, Quu, kff))
        return (Vn, vn), (K, kff, dV)

    _, (Ks, kffs, dVs) = jax.lax.scan(body, (V, v), jnp.arange(N)[::-1])
    rev = lambda z: jnp.swapaxes(z, 0, 1)[:, ::-1]
    return rev(Ks), rev(kffs), dVs.sum(0)


def _forward_ls(e0, du_seq, K, kff, xr_seq, uref_seq, d, par, e_seq):
    """Rollout of all line-search steps at once; returns (B,n_alpha,...)."""
    N = du_seq.shape[1]

    def one(alpha):
        def body(carry, k):
            e, = carry
            dx = e - e_seq[:, k]
            dun = du_seq[:, k] + alpha * kff[:, k] + jnp.einsum("bij,bj->bi", K[:, k], dx)
            en = edyn(e, dun, xr_seq[:, k], xr_seq[:, k + 1], d, par, uref_seq[:, k])
            return (en,), (en, dun)
        (_,), (es, dus) = jax.lax.scan(body, (e0,), jnp.arange(N))
        es = jnp.swapaxes(es, 0, 1)
        return jnp.concatenate([e0[:, None], es], 1), jnp.swapaxes(dus, 0, 1)

    outs = [one(a) for a in ALPHAS]
    return jnp.stack([o[0] for o in outs], 1), jnp.stack([o[1] for o in outs], 1)


def _ilqr_step(du_seq, e0, xr_seq, uref_seq, S, c, P_term, d, par, mu):
    e_seq = rollout_err(e0, du_seq, xr_seq, uref_seq, d, par)
    J0 = traj_cost(e_seq, du_seq, S, c, P_term)
    A, Bm = lin_traj(e0, du_seq, xr_seq, d, par, uref_seq, e_seq)
    K, kff, dV = _backward(A, Bm, S, c, P_term, e_seq, du_seq, mu)
    e_c, du_c = _forward_ls(e0, du_seq, K, kff, xr_seq, uref_seq, d, par, e_seq)
    Js = jax.vmap(lambda ec, dc: traj_cost(ec, dc, S, c, P_term),
                  in_axes=(1, 1), out_axes=1)(e_c, du_c)          # (B,n_alpha)
    Js = jnp.where(jnp.isfinite(Js), Js, jnp.inf)
    best = jnp.argmin(Js, 1)
    Jb = jnp.take_along_axis(Js, best[:, None], 1)[:, 0]
    take = lambda z: jnp.take_along_axis(
        z, best.reshape((-1,) + (1,) * (z.ndim - 1)), 1)[:, 0]
    improved = Jb < J0 - 1e-12
    du_new = jnp.where(improved[:, None, None], take(du_c), du_seq)
    mu_new = jnp.where(improved, jnp.maximum(mu / 10.0, MU_MIN),
                       jnp.minimum(mu * 10.0, MU_MAX))
    return du_new, mu_new


def ilqr_solve(e0, xr_seq, S, c, P_term, d=None, par=None, n_iter=25, n_diff=1,
               uref_seq=None, du0=None):
    """iLQR on (5.1)-(5.2).  Returns du_seq (B,N,4).

    Only the last ``n_diff`` iterations carry gradient.  At a solution
    satisfying LICQ and second-order sufficiency the parameter gradient depends
    on theta only through the Lagrangian *at* the solution, not on the path
    taken there (Buskens & Maurer 2001), so truncating the differentiated
    window costs accuracy only through the residual optimality gap.
    """
    B, N = e0.shape[0], S.shape[1]
    if uref_seq is None:
        uref_seq = uref_from_traj(xr_seq)
    du = jnp.zeros((B, N, NU)) if du0 is None else du0
    mu = jnp.full((B,), 1e-6)
    n_diff = int(min(max(n_diff, 0), n_iter))
    n_frozen = n_iter - n_diff

    if n_frozen > 0:
        Sf, cf, Pf = (jax.lax.stop_gradient(S), jax.lax.stop_gradient(c),
                      jax.lax.stop_gradient(P_term))
        def body(_, carry):
            du_, mu_ = carry
            return _ilqr_step(du_, e0, xr_seq, uref_seq, Sf, cf, Pf, d, par, mu_)
        du, mu = jax.lax.fori_loop(0, n_frozen, body, (du, mu))
        du, mu = jax.lax.stop_gradient(du), jax.lax.stop_gradient(mu)
    for _ in range(n_diff):
        du, mu = _ilqr_step(du, e0, xr_seq, uref_seq, S, c, P_term, d, par, mu)
    return du


def quad_cost_blocks(B, N, Q=None, R=None):
    """A constant hand-weighted stage cost (S_k, c_k) for the NMPC baselines."""
    Q = Q_HAND if Q is None else Q
    R = R_HAND if R is None else R
    S = jnp.zeros((NTAU, NTAU)).at[:NE, :NE].set(jnp.asarray(Q))
    S = S.at[NE:, NE:].set(jnp.asarray(R))
    return (jnp.broadcast_to(S, (B, N, NTAU, NTAU)), jnp.zeros((B, N, NTAU)))


def solve_quad(e0, xr_seq, N, P_term, d=None, par=None, n_iter=25, Q=None, R=None,
               uref_seq=None, return_seq=False):
    """Classical NMPC: hand weights, iLQR, apply the first command.

    Returns ``u`` (B,4) -- the absolute CTBR command -- and an aux dict.
    """
    B = e0.shape[0]
    S, c = quad_cost_blocks(B, N, Q, R)
    if uref_seq is None:
        uref_seq = uref_from_traj(xr_seq)
    du = ilqr_solve(e0, xr_seq, S, c, P_term, d, par, n_iter, 0, uref_seq)
    u = jnp.clip(uref_seq[:, 0] + du[:, 0], jnp.asarray(U_LO), jnp.asarray(U_HI))
    aux = {"du": du, "uref": uref_seq}
    if return_seq:
        aux["e_seq"] = rollout_err(e0, du, xr_seq, uref_seq, d, par)
    return (u, aux) if return_seq else u


# --------------------------------------------------------------------------- #
# §5.12  baselines -- the hover linearisation, LQR, the terminal matrix
# --------------------------------------------------------------------------- #
XI = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

#: Hand-chosen **stage-cost** weights for the LQR and NMPC baselines.  Chosen by
#: measurement, not by taste: the operating point is the argmin of a (Q_pos, R)
#: grid evaluated on the true plant, which is the same grid §8.1 step 6 sweeps
#: and reports.  Over that grid closed-loop RMSE runs from 0.042 m to 1.05 m, a
#: best-to-worst ratio of 25 -- that ratio is the fraction of "controller
#: performance" that is really weight tuning, and is what §8.3 removes.
#: Somebody has to choose these; here that somebody is a documented search.
Q_HAND = np.diag([2.0, 2.0, 2.0, 0.8, 0.8, 0.8, 1.0, 1.0, 0.5])
R_HAND = np.diag([0.5, 1.0, 1.0, 1.0])

#: **Reward** weights of (5.4).  A *different object* from the stage cost above:
#: the stage cost shapes an optimisation the controller solves, the reward
#: scores the closed loop for PPO.  Conflating them is the kind of silent error
#: §11 exists to catch -- the cost weights are ~100x too small to give the
#: policy gradient any signal against the omega and du penalties of (5.4).
Q_REW = np.diag([200.0, 200.0, 200.0, 20.0, 20.0, 20.0, 20.0, 20.0, 10.0])
R_REW = np.diag([2.0, 2.0, 2.0, 2.0])


def _hover_ref():
    q = jnp.asarray([[1.0, 0.0, 0.0, 0.0]])
    return (jnp.concatenate([jnp.asarray([[0.0, 0.0, 1.5]]), jnp.zeros((1, 3)), q], -1),
            jnp.concatenate([jnp.full((1, 1), U_HOVER), jnp.zeros((1, 3))], -1))


def lqr_matrices(check=True):
    """(5.6) continuous error-coordinate hover linearisation (A_c, B_c).

    **Derived by autodiff of (4.1)+(3.3), not pasted** (C-1, C-2).  The
    specification's (5.6) carries ``-g*Xi`` in the velocity/attitude block and
    ``0.5*diag(om_max)`` in the rate block; both are wrong.  With ``delta =
    2 q_ev`` we have ``z_b ~ e3 + delta x e3``, giving ``+g*Xi``; and
    ``delta_dot = 2 q_v_dot = om``, giving ``diag(om_max)``.  Using the spec's
    signs produces an LQR gain whose closed-loop spectral radius on the true
    plant is 1.18 -- unstable.
    """
    xr, ur = _hover_ref()
    edot = lambda e, u: jax.jvp(
        lambda t: err(e_to_state(e, xr) + t * fc(e_to_state(e, xr), u), xr), (0.0,), (1.0,))[1]
    Ac = np.asarray(jax.jacobian(edot, 0)(jnp.zeros((1, NE)), ur))[0, :, 0, :]
    Bc = np.asarray(jax.jacobian(edot, 1)(jnp.zeros((1, NE)), ur))[0, :, 0, :]
    if check:
        A_ref = np.zeros((NE, NE)); A_ref[0:3, 3:6] = np.eye(3); A_ref[3:6, 6:9] = P.g * XI
        B_ref = np.zeros((NE, NU)); B_ref[5, 0] = DAZ_DC_HOVER
        B_ref[6:9, 1:4] = np.diag(OM_MAX)
        assert np.abs(Ac - A_ref).max() < 1e-9, f"A_c drifted:\n{Ac}"
        assert np.abs(Bc - B_ref).max() < 1e-9, f"B_c drifted:\n{Bc}"
    return Ac, Bc


def discretise(Ac, Bc, dt=None):
    """Exact zero-order-hold discretisation by one matrix exponential."""
    from scipy.linalg import expm
    dt = P.dt_c if dt is None else dt
    n, m = Bc.shape
    Mx = np.zeros((n + m, n + m)); Mx[:n, :n] = Ac; Mx[:n, n:] = Bc
    Ed = expm(Mx * dt)
    return Ed[:n, :n], Ed[:n, n:]


def dlqr(Q=None, R=None, dt=None):
    """Offline discrete Riccati solve; returns (K, P_inf, Ad, Bd)."""
    from scipy.linalg import solve_discrete_are
    Q = Q_HAND if Q is None else np.asarray(Q)
    R = R_HAND if R is None else np.asarray(R)
    Ad, Bd = discretise(*lqr_matrices(), dt)
    Pi = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ Pi @ Bd, Bd.T @ Pi @ Ad)
    return K, Pi, Ad, Bd


K_LQR, P_RIC, AD_HOVER, BD_HOVER = dlqr()
PTt = jnp.asarray(P_RIC)       #: default terminal matrix for the NMPC baselines
assert np.abs(np.linalg.eigvals(AD_HOVER - BD_HOVER @ K_LQR)).max() < 1.0, \
    "hover LQR is not stabilising -- check C-1/C-2"


def make_lqr_ctrl(env=None, feedforward=True, K=None):
    """Classical anchor: constant-gain state feedback plus the exact (4.8) term.

    ``feedforward=False`` is kept as a row because without it a state-feedback
    law lags a moving reference for the same reason a frozen-reference MPC does,
    and the comparison would measure the missing term rather than the law.
    """
    K = K_LQR if K is None else K
    Kj = jnp.asarray(K)

    def f(o, e, xr, uref=None, d=None):
        du = -jnp.einsum("ij,bj->bi", Kj, e)
        base = (uref if (feedforward and uref is not None)
                else jnp.zeros_like(du).at[:, 0].set(U_HOVER))
        return jnp.clip(base + du, jnp.asarray(U_LO), jnp.asarray(U_HI)), {}
    f.needs_uref = True
    f.name = "LQR" if feedforward else "LQR (no feed-fwd)"
    return f


def make_nmpc_ctrl(env=None, N=1, Pterm=None, n_iter=25, Q=None, R=None, use_d=False):
    """Classical NMPC: hand weights, iLQR over N stages, preview from the env.

    The solve is jitted once at bind time; re-tracing it every control step is
    both slow and, at N=10, enough to exhaust the compiler.
    """
    Pterm = PTt if Pterm is None else Pterm
    state = {"env": env}

    @jax.jit
    def _solve(e, xr_seq, uref_seq, d):
        Pt = jnp.broadcast_to(Pterm, (e.shape[0], NE, NE))
        return solve_quad(e, xr_seq, N, Pt, d, None, n_iter, Q, R, uref_seq)

    def f(o, e, xr, uref=None, d=None):
        ev = state["env"]
        B = e.shape[0]
        if ev is not None:
            xr_seq, uref_seq = ev.ref_traj(N), ev.ref_useq(N)
        else:
            xr_seq = jnp.broadcast_to(xr[:, None], (B, N + 1, NX))
            uref_seq = uref_from_traj(xr_seq)
        return _solve(e, xr_seq, uref_seq, d if use_d else None), {}

    f.bind_env = lambda ev: state.__setitem__("env", ev)
    f.name = f"NMPC N={N}"
    return f


def make_frozen_nmpc_ctrl(N=1, Pterm=None, n_iter=25):
    """§8.1 step 4: genuinely frozen reference -- deliberately **no** bind_env.

    The frozen controller is handed the instantaneous x_r(t) broadcast over the
    horizon.  ``rollout_eval`` only supplies the preview to a controller that
    exposes ``bind_env``, so the omission is what makes the column frozen; NB1
    asserts the two families of columns differ (T-6).
    """
    Pterm = PTt if Pterm is None else Pterm

    @jax.jit
    def _solve(e, xr):
        B = e.shape[0]
        Pt = jnp.broadcast_to(Pterm, (B, NE, NE))
        xr_seq = jnp.broadcast_to(xr[:, None], (B, N + 1, NX))
        return solve_quad(e, xr_seq, N, Pt, None, None, n_iter,
                          uref_seq=uref_from_traj(xr_seq))

    def f(o, e, xr, uref=None, d=None):
        return _solve(e, xr), {}
    f.name = f"NMPC N={N} (frozen)"
    return f


PID_DEFAULT = dict(kp_pos=(3.0, 3.0, 4.0), kd_pos=(2.4, 2.4, 3.2), ki_pos=(0.4, 0.4, 0.8),
                   k_att=(7.0, 7.0, 2.5), tilt_max=np.deg2rad(35.0))


def make_pid_ctrl(gains=None):
    """Cascaded position -> velocity -> attitude -> rate.

    **Excluded from the main study comparison** (§5.12): it shares no design
    object with the others -- different structure, nine hand gains, a different
    tuning budget -- so its column would measure tuning effort rather than a
    controller class.  It exists only as the ROS incumbent of §9.8.
    """
    g = dict(PID_DEFAULT if gains is None else gains)
    kp, kd, ki = (jnp.asarray(g[k]) for k in ("kp_pos", "kd_pos", "ki_pos"))
    katt, tmax = jnp.asarray(g["k_att"]), g["tilt_max"]
    st = {"I": None}

    def f(o, e, xr, uref=None, d=None):
        B = e.shape[0]
        if st["I"] is None or st["I"].shape[0] != B:
            st["I"] = jnp.zeros((B, 3))
        st["I"] = jnp.clip(st["I"] + e[:, 0:3] * P.dt_c, -2.0, 2.0)
        a_des = -kp * e[:, 0:3] - kd * e[:, 3:6] - ki * st["I"]
        zb_des = a_des + P.g * jnp.asarray(E3)
        n = jnp.linalg.norm(zb_des, axis=-1, keepdims=True)
        zb_des = zb_des / n
        tilt = jnp.arccos(jnp.clip(zb_des[:, 2:3], -1.0, 1.0))
        zb_des = jnp.where(tilt > tmax,
                           qzaxis(qexp_tilt(jnp.cross(jnp.broadcast_to(jnp.asarray(E3),
                                            zb_des.shape), zb_des), jnp.full((B,), tmax))),
                           zb_des)
        q_des = ref_attitude(a_des)
        x = e_to_state(e, xr)
        qe = qmul(qconj(x[:, 6:10]), q_des)
        sgn = jnp.where(qe[:, :1] >= 0.0, 1.0, -1.0)
        om_cmd = katt * 2.0 * sgn * qe[:, 1:4]
        c = jnp.sqrt(jnp.clip(M_TOT * n[:, 0] / T_MAX, 0.0, 1.0))[:, None]
        return jnp.clip(jnp.concatenate([c, om_cmd / jnp.asarray(OM_MAX)], -1),
                        jnp.asarray(U_LO), jnp.asarray(U_HI)), {}
    f.reset = lambda: st.__setitem__("I", None)
    f.name = "PID"
    return f


# --------------------------------------------------------------------------- #
# §5.9  cost map
# --------------------------------------------------------------------------- #
Q_LO, Q_HI = 1.0, 1.0e5          # the learned weights span ~5 orders of magnitude
REPS = ("diag", "chol", "full")
REP_DIM = {"diag": NTAU, "chol": NTAU * (NTAU + 1) // 2, "full": NTAU * NTAU}
_TRIL = np.tril_indices(NTAU)


def mlp_init(key, sizes, scale_last=0.1):
    """Orthogonal-ish MLP init; the head is scaled by 0.1 with zero bias (§5.9)."""
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        s = (scale_last if i == len(sizes) - 2 else np.sqrt(2.0 / a))
        ps.append((jax.random.normal(k, (a, b)) * s, jnp.zeros((b,))))
    return ps


def mlp_apply(params, x, act=jnp.tanh):
    for W, b in params[:-1]:
        x = act(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def costmap_head(theta, obs, rep, N):
    """Trunk + head -> the raw per-stage parameter block, (B,N,REP_DIM[rep])."""
    h = mlp_apply(theta["trunk"], obs)
    z = mlp_apply(theta["head"], h)
    return z.reshape(obs.shape[0], N, REP_DIM[rep])


def costmap_apply(theta, obs, rep, N):
    """-> S (B,N,13,13) PSD, c (B,N,13).

    Initialisation asymmetry, recorded because it matters (§5.9): with a
    0.1-scaled zero-bias head, ``chol``/``full`` start at A ~ 0 so S ~ Q_LO*I --
    the quadratic term is effectively *absent* at init -- whereas ``diag``
    starts at the sigmoid mid-range ~ (Q_LO+Q_HI)/2 = 5e4.  The three do not
    start from comparable places, which predicts the richer forms need
    **longer**, not that they are incapable.
    """
    B = obs.shape[0]
    z = costmap_head(theta, obs, rep, N)
    if rep == "diag":
        dg = Q_LO + (Q_HI - Q_LO) * jax.nn.sigmoid(z)
        S = jnp.einsum("bnij,bnj->bnij", jnp.broadcast_to(jnp.eye(NTAU), (B, N, NTAU, NTAU)), dg)
    elif rep == "chol":
        A = jnp.zeros((B, N, NTAU, NTAU)).at[..., _TRIL[0], _TRIL[1]].set(z)
        dgi = jnp.arange(NTAU)
        A = A.at[..., dgi, dgi].set(jax.nn.softplus(A[..., dgi, dgi]) + 1e-6)
        S = jnp.einsum("bnij,bnkj->bnik", A, A) + Q_LO * jnp.eye(NTAU)
    elif rep == "full":
        A = z.reshape(B, N, NTAU, NTAU)
        S = jnp.einsum("bnij,bnkj->bnik", A, A) + Q_LO * jnp.eye(NTAU)
    else:
        raise ValueError(rep)
    c = theta.get("c_head", None)
    c = (jnp.zeros((B, N, NTAU)) if c is None
         else mlp_apply(c, mlp_apply(theta["trunk"], obs)).reshape(B, N, NTAU))
    return S, c


def costmap_init(key, obs_dim, hid, rep, N, n_layer=2, with_c=False):
    k1, k2, k3 = jax.random.split(key, 3)
    th = {"trunk": mlp_init(k1, [obs_dim] + [hid] * n_layer, scale_last=np.sqrt(2.0 / hid)),
          "head": mlp_init(k2, [hid, N * REP_DIM[rep]], scale_last=0.1)}
    if with_c:
        th["c_head"] = mlp_init(k3, [hid, N * NTAU], scale_last=0.1)
    return th


def mpc_layer(obs, e0, xr_seq, theta, cfg, d=None, uref_seq=None, par=None):
    """The differentiable controller: cost map -> iLQR -> first command."""
    B, N = e0.shape[0], cfg["N"]
    S, c = costmap_apply(theta, obs, cfg["rep"], N)
    Pt = jnp.broadcast_to(cfg.get("P_term", PTt), (B, NE, NE))
    if uref_seq is None:
        uref_seq = uref_from_traj(xr_seq)
    du = ilqr_solve(e0, xr_seq, S, c, Pt, d, par, cfg["n_iter"], cfg.get("n_diff", 1),
                    uref_seq)
    u = jnp.clip(uref_seq[:, 0] + du[:, 0], jnp.asarray(U_LO), jnp.asarray(U_HI))
    return u, {"du": du, "S": S, "c": c}


# --------------------------------------------------------------------------- #
# §5.10  observation and reward
# --------------------------------------------------------------------------- #
OBS_DIM = 40
PREVIEW_STRIDE = 5        # control steps -> lookaheads of 0.1 / 0.2 / 0.3 s
PREVIEW_H = (1, 2, 3)
EMA_TAU = 0.5             # [s] time constant of the running means in (5.3)
NOISE_LEVELS = {
    "off": dict(p=0.0, v=0.0, q=0.0, w=0.0),
    "low": dict(p=0.005, v=0.02, q=0.0035, w=0.01),
    "high": dict(p=0.020, v=0.08, q=0.0140, w=0.04),
}
CRASH_Z, MAX_POS_ERR, MAX_RATE = 0.02, 3.0, 25.0


def build_obs(e, xr, prev, ep, t, s, oracle=None):
    """(5.3), OBS_DIM = 40, plus 6 when the oracle channel is on.

    Preview stride is **5 control steps**, i.e. lookaheads 0.1/0.2/0.3 s; it is
    otherwise invisible.  ``prev`` carries the integral and the running means.
    """
    p = s[..., SP]
    prev_blocks = []
    for h in PREVIEW_H:
        pr, vr, _ = _ref_pva(ep, t + h * PREVIEW_STRIDE * P.dt_c)
        prev_blocks += [pr - p, vr]
    o = jnp.concatenate([e, jnp.concatenate(prev_blocks, -1),
                         jnp.clip(prev["int_ep"], -5.0, 5.0),
                         prev["du_bar"], prev["ev_bar"], prev["om_bar"]], -1)
    if oracle is not None:
        o = jnp.concatenate([o, oracle], -1)
    return o


def reward_quad(e, du, om, d_u, crash, Q=None, R=None):
    """(5.4).  Defaults are Q_REW/R_REW, not the stage weights; see there."""
    Q = Q_REW if Q is None else Q
    R = R_REW if R is None else R
    return (-(jnp.einsum("bi,ij,bj->b", e, jnp.asarray(Q), e)
              + jnp.einsum("bi,ij,bj->b", du, jnp.asarray(R), du)) / 100.0
            - 0.02 * jnp.sum(om ** 2, -1) - 0.05 * jnp.sum(d_u ** 2, -1)
            - 5.0 * crash)


def reward_prog(dk, dk1, om, d_u, crash):
    """(5.5)"""
    return ((dk - dk1) - 0.6 * dk1 ** 2 - 0.02 * jnp.sum(om ** 2, -1)
            - 0.05 * jnp.sum(d_u ** 2, -1) + 0.05 - 5.0 * crash)


# --------------------------------------------------------------------------- #
# §5.11  environment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EnvCfg:
    """Hashable -> usable as a JIT static argument; two Envs with an equal cfg
    share one compiled program."""
    n: int = 16
    ep_len: int = 250
    paths: tuple = ("circle",)
    ep_kind: str = "sample"
    task: str = "track"            # 'track' | 'stabilize'
    noise: str = "off"             # 'off' | 'low' | 'high'
    oracle: bool = False
    oracle_target: str = "wrench"  # 'wrench' | 'residual'
    mixer: str = "nominal"         # 'nominal' | 'true'
    reward: str = "quad"           # 'quad' | 'prog'
    dmod_mode: str = "first_order"


def nominal_spec(**kw):
    """S1: the nominal plant.  Speeds still vary; nothing else does."""
    s = dict(speed=(0.5, 1.5), wind=(0.0, 0.0), m=(1.0, 1.0), D=(1.0, 1.0),
             tau=(1.0, 1.0), T=(1.0, 1.0), Kw=(1.0, 1.0), J=(1.0, 1.0))
    s.update(kw); return s


def disturbed_spec(**kw):
    """S2 **raw**, straight from §7.2.  Moderate it before evaluating."""
    s = dict(speed=(0.5, 4.0), wind=(0.0, 15.0), m=(0.80, 1.20), D=(0.50, 2.00),
             tau=(0.50, 3.00), T=(0.85, 1.15), Kw=(0.60, 1.40), J=(0.70, 1.40))
    s.update(kw); return s


def ood_spec(**kw):
    """S3 **raw**, straight from §7.2.  Moderate it before evaluating."""
    s = dict(speed=(5.0, 6.5), wind=(18.0, 25.0), m=(0.65, 1.35), D=(2.20, 3.00),
             tau=(3.50, 5.00), T=(0.75, 1.30), Kw=(0.40, 1.80), J=(0.50, 1.80))
    s.update(kw); return s


def _u(key, lohi, shape):
    lo, hi = lohi
    return jax.random.uniform(key, shape, minval=float(lo), maxval=float(hi))


def wind_at(ep, t):
    """(3.9) per-episode wind, deterministic in t."""
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1, 1)
    return ep["wbar"][:, None] * (
        (1.0 + 0.35 * jnp.sin(2 * jnp.pi * 0.23 * t + ep["wphi"][:, None])) * ep["wdir"]
        + 0.35 * jnp.sin(2 * jnp.pi * 0.7 * t) * ep["wperp"])


def _ep_where(mask, new, old):
    """Select per-vehicle between two episode/param pytrees."""
    return {k: jnp.where(mask.reshape((-1,) + (1,) * (jnp.ndim(v) - 1)), new[k], v)
            for k, v in old.items()}


class Env:
    """Batched quadrotor environment: ``n`` vehicles advanced as one array program.

    The transition is a single jitted pure function, so two Envs with an equal
    :class:`EnvCfg` share one compiled program.  Episode termination is
    per-vehicle: a vehicle that crashes, drifts past the 3 m bound, spins up
    past 25 rad/s or simply reaches ``ep_len`` is resampled in place and counted
    in ``respawns``.

    > The 3 m bound is a measurement trap (§5.10).  Under a severe disturbance
    > every controller terminates at the bound and every RMSE converges to the
    > same value.  Always report the saturation fraction beside RMSE.
    """

    def __init__(self, n, seed, ep_len, dist, paths, ep_kind=None, noise="off",
                 oracle=False, mixer="nominal", fixed=None, task="track",
                 reward="quad", oracle_target="wrench", scen=None, level=0,
                 dmod_mode="first_order"):
        self.cfg = EnvCfg(n=n, ep_len=ep_len, paths=tuple(paths),
                          ep_kind=ep_kind or "sample", task=task, noise=noise,
                          oracle=oracle, oracle_target=oracle_target, mixer=mixer,
                          reward=reward, dmod_mode=dmod_mode)
        self.n, self.dist = n, dict(dist)
        self.fixed = dict(fixed or {})
        self.key = jax.random.PRNGKey(int(seed))
        self.obs_dim = OBS_DIM + (6 if oracle else 0)
        self.scen, self.level = scen, level
        self.estimator = None
        self.respawns = 0
        self.no_respawn = False        # §7.3 video clips pin this True
        self.reset()

    # -- episode sampling --------------------------------------------------- #
    def _sample_ep(self, key, n):
        ks = jax.random.split(key, 20)
        pidx = jnp.asarray([PATH_IDX[p] for p in self.cfg.paths])
        kind = pidx[jax.random.randint(ks[0], (n,), 0, len(pidx))]
        c = jnp.stack([_u(ks[1], (-1.5, 1.5), (n,)), _u(ks[2], (-1.5, 1.5), (n,)),
                       _u(ks[3], (1.0, 2.5), (n,))], -1)
        R = _u(ks[4], (0.5, 2.0), (n,))
        spd = _u(ks[6], self.dist["speed"], (n,))
        omega = jnp.take_along_axis(
            jnp.stack([path_omega(i, R, spd) for i in range(len(PATHS_ALL))], -1),
            kind[:, None], 1)[:, 0]
        az = _u(ks[7], (0.0, 2 * np.pi), (n,))
        ep = dict(kind=kind, c=c, R=R, phi0=_u(ks[5], (0.0, 2 * np.pi), (n,)),
                  omega=omega, spd=spd,
                  delta=_u(ks[8], (-1.0, 1.0), (n, 3)) * jnp.asarray([1.0, 1.0, 0.5]),
                  wbar=_u(ks[9], self.dist["wind"], (n,)),
                  wphi=_u(ks[10], (0.0, 2 * np.pi), (n,)),
                  wdir=jnp.stack([jnp.cos(az), jnp.sin(az), jnp.zeros(n)], -1),
                  wperp=jnp.stack([-jnp.sin(az), jnp.cos(az), jnp.zeros(n)], -1),
                  e0=_u(ks[11], (-0.5, 0.5), (n, 3)))
        for i, k in enumerate(("m", "D", "tau", "T", "Kw", "J")):
            ep[f"lam_{k}"] = _u(jax.random.fold_in(ks[12], i), self.dist[k], (n,))
        if self.scen is not None:
            ep.update(self.scen.sample(ks[13], n, self.level))
        for k, v in self.fixed.items():
            if k in ep:
                ep[k] = jnp.broadcast_to(jnp.asarray(v, dtype=jnp.float64), jnp.shape(ep[k]))
        return ep

    def _make_par(self, ep, n):
        kw = dict(m_scale=ep["lam_m"], D_scale=ep["lam_D"], tau_scale=ep["lam_tau"],
                  Tmax_scale=ep["lam_T"], Kw_scale=ep["lam_Kw"], J_scale=ep["lam_J"])
        if self.scen is not None:
            kw.update(self.scen.par_kwargs(ep, n))
        par = make_par(n, **kw)
        if self.cfg.mixer == "true":
            Mt = jnp.concatenate([jnp.ones((n, 1, 4)),
                                  par["Mtau_w2"] / par["KT"][:, :, None]], 1)
            par = {**par, "Minv_ctrl": jnp.linalg.inv(Mt), "J_ctrl": par["J"]}
        return par

    def _init(self, ep, n):
        xr, _, _ = ref_state(ep, jnp.zeros(n))
        par = self._make_par(ep, n)
        Om = jnp.broadcast_to(jnp.sqrt(par["m"] * P.g / 4.0 / par["KT"]), (n, 4))
        z = jnp.zeros((n, 3))
        s = jnp.concatenate([xr[:, 0:3] + ep["e0"], xr[:, 6:10], xr[:, 3:6], z, Om, z, z], -1)
        prev = dict(int_ep=z, du_bar=jnp.zeros((n, NU)), ev_bar=z, om_bar=z,
                    u=hover_u(n, par))
        return s, par, prev

    def reset(self, key=None):
        self.key, k = jax.random.split(self.key if key is None else key)
        self.ep = self._sample_ep(k, self.n)
        self.state, self.par, self.prev = self._init(self.ep, self.n)
        self.t = jnp.zeros(self.n)
        self.n_step = jnp.zeros(self.n)
        self.respawns = 0
        return self.obs()

    # -- accessors ---------------------------------------------------------- #
    def ref_now(self):
        return ref_state(self.ep, self.t)

    def ref_traj(self, N):
        """(B,N+1,10): the preview that (5.1) scores against."""
        return _ref_traj(self.ep, self.t, N)

    def ref_useq(self, N):
        """(B,N,4): the **exact analytic** u_ref over the horizon (N-3)."""
        return _ref_useq(self.ep, self.t, N)

    def wind(self):
        return wind_at(self.ep, self.t)

    def sdot(self, u=None):
        return fp(self.state, self.prev["u"] if u is None else u, self.par, self.wind())

    def d_truth(self):
        """The ground-truth wrench [N, N m].  Logger and evaluator only."""
        return _wrench_jit(self.state, self.prev["u"], self.par, self.wind())

    def d_channel(self):
        """**RAW wrench** -- the observation channel only (§4.4).

        With an estimator attached this returns the *prediction*, which is the
        whole point of ``AdaptEnv``; without one it returns the truth.
        """
        if self.estimator is not None:
            return self.estimator(self)
        return self.d_truth()

    def dmod(self):
        """(4.12)+(4.13) converted residual [m/s^2, rad/s] -- the **model** only."""
        return wrench_to_dmod(self.d_channel(), self.cfg.dmod_mode)

    def residual_truth(self):
        """(4.9) the true model residual."""
        return _resid_jit(self.state, self.prev["u"], self.par, self.wind())

    def _oracle(self):
        if not self.cfg.oracle:
            return None
        return self.d_truth() if self.cfg.oracle_target == "wrench" else self.residual_truth()

    def obs(self):
        self.key, k = jax.random.split(self.key)
        o, e, xr = _obs_jit(self.cfg, self.state, self.ep, self.prev, self.t, k)
        if self.cfg.oracle:
            o = jnp.concatenate([o, self._oracle()], -1)
        return o, e, xr

    def frame26(self):
        """(6.2) the 26-D RDP input frame for the current state."""
        _, uref, _ = self.ref_now()
        return _frame26(self.state, self.ep, self.t, self.prev["u"], uref, self.par)

    # -- transition --------------------------------------------------------- #
    def step(self, u):
        self.key, k1, k2 = jax.random.split(self.key, 3)
        ep_new = self._sample_ep(k1, self.n)
        s_new, par_new, prev_new = self._init(ep_new, self.n)
        (self.state, self.par, self.ep, self.prev, self.t, self.n_step,
         r, done, info, nres) = _step_jit(
            self.cfg, self.no_respawn, self.state, self.par, self.ep, self.prev,
            self.t, self.n_step, u, s_new, par_new, ep_new, prev_new)
        self.respawns += int(nres)
        return r, done, info


# --------------------------------------------------------------------------- #
# jitted kernels behind Env
# --------------------------------------------------------------------------- #
@functools.partial(jax.jit, static_argnums=(2,))
def _ref_traj(ep, t, N):
    return jnp.stack([ref_state(ep, t + k * P.dt_c)[0] for k in range(N + 1)], 1)


@functools.partial(jax.jit, static_argnums=(2,))
def _ref_useq(ep, t, N):
    return jnp.stack([ref_state(ep, t + k * P.dt_c)[1] for k in range(N)], 1)


_wrench_jit = jax.jit(lambda s, u, par, vw: external_wrench(s, u, par, fp(s, u, par, vw)))
_resid_jit = jax.jit(lambda s, u, par, vw: true_disturbance(s, u, par, vw))


@functools.partial(jax.jit, static_argnums=(0,))
def _obs_jit(cfg, state, ep, prev, t, key):
    xr, _, _ = ref_state(ep, t)
    e = err(plant_to_ctrl(state), xr)
    lv = NOISE_LEVELS[cfg.noise]
    if any(v > 0 for v in lv.values()):
        ks = jax.random.split(key, 3)
        n = state.shape[0]
        e = e + jnp.concatenate([lv["p"] * jax.random.normal(ks[0], (n, 3)),
                                 lv["v"] * jax.random.normal(ks[1], (n, 3)),
                                 lv["q"] * jax.random.normal(ks[2], (n, 3))], -1)
    return build_obs(e, xr, prev, ep, t, state), e, xr


@functools.partial(jax.jit, static_argnums=(0, 1))
def _step_jit(cfg, no_respawn, state, par, ep, prev, t, n_step, u,
              s_new, par_new, ep_new, prev_new):
    u = jnp.clip(u, jnp.asarray(U_LO), jnp.asarray(U_HI))
    xr, uref, _ = ref_state(ep, t)
    e = err(plant_to_ctrl(state), xr)
    d_u, du = u - prev["u"], u - uref
    s_next = step_p(state, u, par, wind_at(ep, t))
    t_next = t + P.dt_c
    xr_n, _, _ = ref_state(ep, t_next)
    e_n = err(plant_to_ctrl(s_next), xr_n)

    bad = (~jnp.isfinite(s_next).all(-1)) | (s_next[:, 2] <= CRASH_Z)
    far = jnp.linalg.norm(e_n[:, 0:3], axis=-1) > MAX_POS_ERR
    spin = jnp.linalg.norm(s_next[:, SW], axis=-1) > MAX_RATE
    crash = (bad | spin).astype(jnp.float64)
    done = bad | far | spin
    om = s_next[:, SW]
    r = (reward_prog(jnp.linalg.norm(e[:, 0:3], axis=-1),
                     jnp.linalg.norm(e_n[:, 0:3], axis=-1), om, d_u, crash)
         if cfg.reward == "prog" else reward_quad(e_n, du, om, d_u, crash))

    a = float(np.exp(-P.dt_c / EMA_TAU))
    prev_next = dict(
        int_ep=jnp.clip(prev["int_ep"] + e_n[:, 0:3] * P.dt_c, -5.0, 5.0),
        du_bar=a * prev["du_bar"] + (1 - a) * du,
        ev_bar=a * prev["ev_bar"] + (1 - a) * e_n[:, 3:6],
        om_bar=a * prev["om_bar"] + (1 - a) * om, u=u)
    n_step = n_step + 1.0
    sat_hi = (u[:, 0] >= U_HI[0] - 1e-9)
    info = dict(e=e_n, sat=jnp.mean(((u[:, 0] <= U_LO[0] + 1e-9) | sat_hi).astype(jnp.float64)),
                sat_v=((u[:, 0] <= U_LO[0] + 1e-9) | sat_hi).astype(jnp.float64),
                crash=crash, pos_err=jnp.linalg.norm(e_n[:, 0:3], axis=-1),
                tilt=jnp.linalg.norm(e_n[:, 6:9], axis=-1), u=u, du=du, om=om,
                done=done)

    reset = (done | (n_step >= cfg.ep_len)) & (not no_respawn)
    info["reset"] = reset
    m = reset[:, None]
    return (jnp.where(m, s_new, s_next), _ep_where(reset, par_new, par),
            _ep_where(reset, ep_new, ep), _ep_where(reset, prev_new, prev_next),
            jnp.where(reset, 0.0, t_next), jnp.where(reset, 0.0, n_step),
            r, done, info, jnp.sum(reset))


def _frame26(state, ep, t, u_prev, uref, par):
    """(6.2) 26-D causal frame: [p-p_r | vec(R) row-major | v | om || u-u_ref || PWM]."""
    pr, _, _ = _ref_pva(ep, t)
    R = qrotmat(state[..., SQ]).reshape(state.shape[0], 9)
    _, pwm, _ = inner_loop(state, u_prev, par)
    return jnp.concatenate([state[..., SP] - pr, R, state[..., SV], state[..., SW],
                            u_prev - uref, pwm], -1)


# --------------------------------------------------------------------------- #
# §5.14  evaluation and I/O
# --------------------------------------------------------------------------- #
def apath(*parts):
    """Path inside the artefact tree, creating the directory."""
    p = os.path.join(ARTIFACTS, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def save_ckpt(obj, *parts):
    p = apath(*parts)
    with open(p, "wb") as fh:
        pickle.dump(obj, fh)
    return p


def load_ckpt(*parts):
    p = apath(*parts)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def save_json(obj, *parts):
    p = apath(*parts)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: (
            o.tolist() if hasattr(o, "tolist") else str(o)))
    return p


def load_json(*parts):
    p = apath(*parts)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def banner(msg, ch="="):
    print(ch * 78); print(msg); print(ch * 78)


def style():
    """Matplotlib defaults.  Colour is never the only carrier of meaning (§10)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.6, "figure.constrained_layout.use": True})


def ctrl_from_actor(actor, cfg, env=None, dmod_fn=None):
    """Wrap a trained cost map into the ``f(o, e, xr) -> (u, aux)`` interface.

    ``dmod_fn`` supplies the model correction of variants B/C (§6.3); when it is
    None the prediction dynamics stay nominal (variant A).  The MPC layer is
    jitted once, so ``solve_latency_ms`` measures the solve and not the tracer.
    """
    state = {"env": env}
    N = cfg["N"]

    @jax.jit
    def _solve(o, e, xr_seq, uref_seq, d):
        return mpc_layer(o, e, xr_seq, actor, cfg, d, uref_seq)[0]

    def f(o, e, xr, uref=None, d=None):
        ev = state["env"]
        if ev is not None:
            xr_seq, uref_seq = ev.ref_traj(N), ev.ref_useq(N)
        else:
            xr_seq = jnp.broadcast_to(xr[:, None], (e.shape[0], N + 1, NX))
            uref_seq = uref_from_traj(xr_seq)
        dm = dmod_fn(ev) if (dmod_fn is not None and ev is not None) else d
        return _solve(o, e, xr_seq, uref_seq, dm), {}

    f.bind_env = lambda ev: state.__setitem__("env", ev)
    f.name = cfg.get("name", "AC-MPC")
    return f


def rollout_eval(env, ctrl, T, warmup=0, record=False):
    """Roll a controller out and return per-vehicle arrays plus timings.

    ``ms_per_step`` here is **throughput** over the whole batch and must never
    be written into a column labelled latency; use :func:`solve_latency_ms`.
    """
    if hasattr(ctrl, "bind_env"):
        ctrl.bind_env(env)
    if hasattr(ctrl, "reset"):
        ctrl.reset()
    o, e, xr = env.obs()
    pe, tl, ef, sm, st, cr, us, rec = [], [], [], [], [], [], [], []
    u_prev = None
    t0 = time.perf_counter()
    for k in range(T):
        _, uref, _ = env.ref_now()
        out = ctrl(o, e, xr, uref=uref)
        u = out[0] if isinstance(out, tuple) else out
        if record:
            rec.append(dict(p=np.asarray(env.state[:, SP]), q=np.asarray(env.state[:, SQ]),
                            pr=np.asarray(xr[:, 0:3]), u=np.asarray(u), t=np.asarray(env.t),
                            om=np.asarray(env.state[:, SW])))
        r, done, info = env.step(u)
        o, e, xr = env.obs()
        if k >= warmup:
            pe.append(np.asarray(info["pos_err"])); tl.append(np.asarray(info["tilt"]))
            ef.append(np.asarray(info["u"][:, 0]) ** 2)
            sm.append(np.zeros(env.n) if u_prev is None
                      else np.asarray(jnp.sum((u - u_prev) ** 2, -1)))
            st.append(np.asarray(info["sat_v"])); cr.append(np.asarray(info["crash"]))
            us.append(np.asarray(info["u"]))
        u_prev = u
    jax.block_until_ready(o)
    wall = time.perf_counter() - t0
    out = dict(pos_err=np.array(pe), tilt=np.array(tl), effort=np.array(ef),
               smooth=np.array(sm), sat=np.array(st), crash=np.array(cr),
               u=np.array(us), ms_per_step=1e3 * wall / max(T, 1),
               respawns=env.respawns, n=env.n, T=T)
    if record:
        out["rec"] = rec
    return out


def stats(r):
    """Summary of a :func:`rollout_eval` result, one row of any table."""
    pe = r["pos_err"]
    per_veh = np.sqrt((pe ** 2).mean(0))
    q1, q3 = np.percentile(per_veh, [25, 75])
    return dict(rmse=float(np.sqrt((pe ** 2).mean())), rmse_iqr=float(q3 - q1),
                maxerr=float(pe.max()), tilt=float(r["tilt"].mean()),
                effort=float(r["effort"].mean()), smooth=float(r["smooth"].mean()),
                sat=float(r["sat"].mean()), crash=float(r["crash"].mean()),
                ms_per_step=float(r["ms_per_step"]), respawns=int(r["respawns"]))


def solve_latency_ms(ctrl, env1, T=40):
    """Single-vehicle solve latency.  The **only** quantity comparable to 20 ms.

    ``ms_per_step / batch`` is throughput and has nothing to do with what one
    aircraft experiences; mixing them produces an apparent 200x speed-up for the
    learned controllers.
    """
    assert env1.n == 1, "latency must be measured on a batch of ONE"
    if hasattr(ctrl, "bind_env"):
        ctrl.bind_env(env1)
    o, e, xr = env1.obs()
    _, uref, _ = env1.ref_now()
    out = ctrl(o, e, xr, uref=uref)
    jax.block_until_ready(out[0] if isinstance(out, tuple) else out)   # compile
    ts = []
    for _ in range(T):
        o, e, xr = env1.obs()
        _, uref, _ = env1.ref_now()
        t0 = time.perf_counter()
        out = ctrl(o, e, xr, uref=uref)
        u = out[0] if isinstance(out, tuple) else out
        jax.block_until_ready(u)
        ts.append(1e3 * (time.perf_counter() - t0))
        env1.step(u)
    return {"median": float(np.median(ts)), "p95": float(np.percentile(ts, 95)),
            "mean": float(np.mean(ts)), "max": float(np.max(ts))}


# --------------------------------------------------------------------------- #
# §5.13  policy optimisation -- PPO / TRPO with GAE and MPVE
# --------------------------------------------------------------------------- #
PPO_DEFAULTS = dict(gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.0, vf_coef=0.5,
                    max_grad_norm=0.5, lr=3e-4, epochs=10, minib=32, sigma=0.15,
                    algo="ppo", mpve=False, kl_target=0.01, rep="diag", N=1,
                    n_iter=5, n_diff=2, hid=256, sat_gate=0.05)


def gae(rew, val, val_last, done, gamma, lam):
    """(5.8) generalised advantage estimation.  rew/val (T,B)."""
    T = rew.shape[0]
    adv = jnp.zeros_like(rew)
    nxt = val_last
    last = jnp.zeros_like(val_last)

    def body(carry, k):
        nxt, last = carry
        nonterm = 1.0 - done[k]
        delta = rew[k] + gamma * nxt * nonterm - val[k]
        last = delta + gamma * lam * nonterm * last
        return (val[k], last), last

    _, out = jax.lax.scan(body, (nxt, last), jnp.arange(T)[::-1])
    return out[::-1]


def mpve_targets(e_seq, du_seq, critic, gamma, om, d_u):
    """(5.9) model-predictive value expansion.

    The MPC already produced a predicted trajectory, so price it: the rollout is
    a by-product of the control computation and costs nothing extra.
    """
    N = du_seq.shape[1]
    tot = jnp.zeros(e_seq.shape[0])
    for k in range(N):
        r = reward_quad(e_seq[:, k + 1], du_seq[:, k], om, d_u, jnp.zeros(e_seq.shape[0]))
        tot = tot + (gamma ** k) * r
    return tot + (gamma ** N) * mlp_apply(critic, e_seq[:, -1])[:, 0]


def critic_init(key, obs_dim, hid, n_layer=2):
    return mlp_init(key, [obs_dim] + [hid] * n_layer + [1], scale_last=0.01)


def _logp(a, mu, log_sigma):
    z = (a - mu) / jnp.exp(log_sigma)
    return jnp.sum(-0.5 * z ** 2 - log_sigma - 0.5 * np.log(2 * np.pi), -1)


def train_ppo(env_fn, cfg, seed=0, actor=None, critic=None, log_path=None,
              iters=100, T_rollout=64, verbose=True, env=None):
    """PPO (Schulman 2017) over the differentiable MPC layer.

    The action is the **cost map**: the policy perturbs the stage weights, the
    iLQR turns them into a command, and the gradient reaches theta through the
    last ``n_diff`` solver iterations.

    ``sat`` is a **gate, not a diagnostic** (§5.13): if collective saturation
    exceeds ``cfg['sat_gate']`` over the last 20 % of iterations a loud warning
    is printed naming the disturbance magnitude.  A policy pinned against its
    input box never learns a disturbance-response manifold, and every
    downstream comparison built on it is void.
    """
    import optax
    import pandas as pd
    c = dict(PPO_DEFAULTS); c.update(cfg)
    env = env_fn() if env is None else env
    key = jax.random.PRNGKey(seed)
    key, ka, kc = jax.random.split(key, 3)
    obs_dim = env.obs_dim
    if actor is None:
        actor = costmap_init(ka, obs_dim, c["hid"], c["rep"], c["N"])
    if critic is None:
        critic = critic_init(kc, obs_dim, c["hid"])
    log_sigma = jnp.full((REP_DIM[c["rep"]] * c["N"],), np.log(c["sigma"]))
    params = {"actor": actor, "critic": critic}
    opt = optax.chain(optax.clip_by_global_norm(c["max_grad_norm"]),
                      optax.adam(c["lr"]))
    opt_state = opt.init(params)
    mcfg = {"N": c["N"], "rep": c["rep"], "n_iter": c["n_iter"], "n_diff": c["n_diff"]}

    def act_mu(actor_p, o):
        return costmap_head(actor_p, o, c["rep"], c["N"]).reshape(o.shape[0], -1)

    def u_from_z(actor_p, o, e, xr_seq, uref_seq, z, d):
        """Apply a *perturbed* cost map: z is the sampled head output."""
        th = dict(actor_p)
        B = o.shape[0]
        S, cc = _costmap_from_z(z.reshape(B, c["N"], REP_DIM[c["rep"]]), c["rep"])
        Pt = jnp.broadcast_to(PTt, (B, NE, NE))
        du = ilqr_solve(e, xr_seq, S, cc, Pt, d, None, c["n_iter"], c["n_diff"], uref_seq)
        return jnp.clip(uref_seq[:, 0] + du[:, 0], jnp.asarray(U_LO), jnp.asarray(U_HI)), du

    rows = []
    for it in range(iters):
        O, Z, LP, RW, VL, DN = [], [], [], [], [], []
        sat_acc, crash_acc, ep_len_acc = [], [], []
        o, e, xr = env.obs()
        for t in range(T_rollout):
            key, k = jax.random.split(key)
            mu = act_mu(params["actor"], o)
            z = mu + jnp.exp(log_sigma) * jax.random.normal(k, mu.shape)
            lp = _logp(z, mu, log_sigma)
            xr_seq, uref_seq = env.ref_traj(c["N"]), env.ref_useq(c["N"])
            d = env.dmod() if c.get("use_d", False) else None
            u, _ = u_from_z(params["actor"], o, e, xr_seq, uref_seq, z, d)
            v = mlp_apply(params["critic"], o)[:, 0]
            r, done, info = env.step(u)
            O.append(o); Z.append(z); LP.append(lp); RW.append(r); VL.append(v)
            DN.append(done.astype(jnp.float64))
            sat_acc.append(float(info["sat"])); crash_acc.append(float(jnp.mean(info["crash"])))
            o, e, xr = env.obs()
        v_last = mlp_apply(params["critic"], o)[:, 0]
        O, Z, LP = jnp.stack(O), jnp.stack(Z), jnp.stack(LP)
        RW, VL, DN = jnp.stack(RW), jnp.stack(VL), jnp.stack(DN)
        ADV = gae(RW, VL, v_last, DN, c["gamma"], c["lam"])
        RET = ADV + VL
        flat = lambda z: z.reshape((-1,) + z.shape[2:])
        Of, Zf, LPf, ADVf, RETf = map(flat, (O, Z, LP, ADV, RET))
        ADVf = (ADVf - ADVf.mean()) / (ADVf.std() + 1e-8)

        def loss_fn(p, ob, zz, lp_old, adv, ret):
            mu = act_mu(p["actor"], ob)
            lp = _logp(zz, mu, log_sigma)
            ratio = jnp.exp(lp - lp_old)
            l_pi = -jnp.mean(jnp.minimum(
                ratio * adv, jnp.clip(ratio, 1 - c["clip"], 1 + c["clip"]) * adv))
            v = mlp_apply(p["critic"], ob)[:, 0]
            l_v = jnp.mean((v - ret) ** 2)
            ent = jnp.mean(jnp.sum(log_sigma + 0.5 * np.log(2 * np.pi * np.e)))
            loss = l_pi + c["vf_coef"] * l_v - c["ent_coef"] * ent
            kl = jnp.mean(lp_old - lp)
            cf = jnp.mean((jnp.abs(ratio - 1.0) > c["clip"]).astype(jnp.float64))
            return loss, (l_pi, l_v, ent, kl, cf)

        grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        n = Of.shape[0]
        mb = max(n // c["minib"], 1)
        aux_last = None
        gnorm = 0.0
        for _ in range(c["epochs"]):
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, n)
            for i in range(c["minib"]):
                idx = perm[i * mb:(i + 1) * mb]
                if idx.size == 0:
                    continue
                (loss, aux), g = grad_fn(params, Of[idx], Zf[idx], LPf[idx],
                                         ADVf[idx], RETf[idx])
                upd, opt_state = opt.update(g, opt_state, params)
                params = optax.apply_updates(params, upd)
                aux_last = aux
                gnorm = float(optax.global_norm(g))
            if aux_last is not None and float(aux_last[3]) > 4 * c["kl_target"]:
                break                                   # early stop on KL

        rows.append(dict(iter=it, reward=float(RW.mean()), ep_len=float(T_rollout),
                         value_loss=float(aux_last[1]), policy_loss=float(aux_last[0]),
                         entropy=float(aux_last[2]), kl=float(aux_last[3]),
                         clipfrac=float(aux_last[4]), grad_norm=gnorm,
                         sat=float(np.mean(sat_acc)), crash_rate=float(np.mean(crash_acc)),
                         wall_s=time.time()))
        if verbose and (it % max(iters // 10, 1) == 0 or it == iters - 1):
            r_ = rows[-1]
            print(f"  it {it:4d}  R {r_['reward']:+8.3f}  vloss {r_['value_loss']:8.3f} "
                  f"kl {r_['kl']:.4f}  sat {r_['sat']:.3f}  crash {r_['crash_rate']:.3f}")
    df = pd.DataFrame(rows)
    df["wall_s"] = df["wall_s"] - df["wall_s"].iloc[0]
    tail = df.iloc[int(0.8 * len(df)):]
    if len(tail) and tail["sat"].mean() > c["sat_gate"]:
        print("!" * 78)
        print(f"!! SATURATION GATE FAILED: collective saturated "
              f"{100*tail['sat'].mean():.1f}% over the last 20% of iterations "
              f"(limit {100*c['sat_gate']:.0f}%).")
        print(f"!! Disturbance magnitude in force: {c.get('dist_label','<unlabelled>')}.")
        print("!! A policy pinned against its input box never learns a "
              "disturbance-response manifold; every downstream comparison built "
              "on it is void.  Reduce the wrench randomisation and retrain.")
        print("!" * 78)
    if log_path:
        df.to_csv(apath(*log_path) if isinstance(log_path, (list, tuple)) else log_path,
                  index=False)
    return params["actor"], params["critic"], df


def _costmap_from_z(z, rep):
    """Shared by training and inference: raw head output -> (S, c)."""
    B, N = z.shape[0], z.shape[1]
    if rep == "diag":
        dg = Q_LO + (Q_HI - Q_LO) * jax.nn.sigmoid(z)
        S = jnp.einsum("bnij,bnj->bnij",
                       jnp.broadcast_to(jnp.eye(NTAU), (B, N, NTAU, NTAU)), dg)
    elif rep == "chol":
        A = jnp.zeros((B, N, NTAU, NTAU)).at[..., _TRIL[0], _TRIL[1]].set(z)
        di = jnp.arange(NTAU)
        A = A.at[..., di, di].set(jax.nn.softplus(A[..., di, di]) + 1e-6)
        S = jnp.einsum("bnij,bnkj->bnik", A, A) + Q_LO * jnp.eye(NTAU)
    else:
        A = z.reshape(B, N, NTAU, NTAU)
        S = jnp.einsum("bnij,bnkj->bnik", A, A) + Q_LO * jnp.eye(NTAU)
    return S, jnp.zeros((B, N, NTAU))
