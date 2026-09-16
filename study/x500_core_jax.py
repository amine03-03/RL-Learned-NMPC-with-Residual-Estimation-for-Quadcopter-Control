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
    K_T: float = 8.54858e-06               # SDF motorConstant
    k_m: float = 0.016                     # SDF momentConstant -- the TRUTH
    Om_max: float = 1000.0                 # SDF maxRotVelocity / SIM_GZ_EC_MAX
    #: SIM_GZ_EC_MIN from the gz_x500 airframe file.  PX4 maps a normalised
    #: actuator command onto [Om_min, Om_max], **not** onto [0, Om_max] (A2).
    #: Ignoring it puts u_hover 5.6 % and d a_z/d c 17.7 % out against the real
    #: platform, and removes the 0.77 N idle floor entirely.
    Om_min: float = 150.0
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
    #: CA_ROTORn_KM from the gz_x500 airframe file -- what the PX4 **allocator**
    #: believes.  It disagrees with the Gazebo plugin's momentConstant (0.016) by
    #: 3.125x, so a commanded yaw torque realises only 32 % of its intent on the
    #: real stack (A3).  Carried separately so that error is represented.
    k_m_ctrl: float = 0.05
    # --- from the SDF, previously unmodelled (A5) ---------------------------
    #: rotorDragCoefficient: a rotor-speed-proportional drag on the airspeed
    #: component perpendicular to the rotor axis, per rotor.
    c_rotor_drag: float = 8.06428e-05
    #: rollingMomentCoefficient: moment opposing in-plane airspeed.
    c_roll_mom: float = 1.0e-06
    #: rotor own inertia (SDF, diagonal, body axes).  Previously the rotors were
    #: treated as point masses, which understated J by up to 0.44 % (A6).
    J_rotor: tuple = (3.8464910483993325e-07, 2.6115851691700804e-05,
                      2.649858234714004e-05)
    # --- declared modelling additions, NOT from the SDF (§2.1) ---------------
    D: tuple = (0.30, 0.30, 0.35)          # bulk translational drag [N s/m]
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


def omega_of_cmd(c):
    """Normalised actuator command -> rotor speed, the PX4 gz_x500 mapping (A2).

        Omega = Om_min + c (Om_max - Om_min),     c in [0, 1]

    **Not** ``c * Om_max``.  SIM_GZ_EC_MIN = 150 rad/s is an idle floor: at
    c = 0 the rotors still make 0.77 N, 3.8 % of weight.
    """
    return P.Om_min + c * (P.Om_max - P.Om_min)


def cmd_of_omega(om):
    """Inverse of :func:`omega_of_cmd`; also the ``actuator_motors`` scaling."""
    return (om - P.Om_min) / (P.Om_max - P.Om_min)


def _derive() -> dict:
    """Everything in §2.2-§2.4, computed.  Nothing here may be pasted."""
    m = P.m_body + P.n_rotor * P.m_rotor                                  # (2.1)
    cg_nom = P.m_rotor * R_ROTOR.sum(0) / m                               # (2.2)
    # (2.3) with the rotors' OWN inertia, not as point masses (A6)
    J_nom = np.diag(P.J_body) + sum(
        np.diag(P.J_rotor) + _inertia_shift(P.m_rotor, r - cg_nom) for r in R_ROTOR)
    f_max = P.K_T * P.Om_max ** 2                                         # (2.4)
    T_max = P.n_rotor * f_max
    f_idle = P.K_T * P.Om_min ** 2
    T_idle = P.n_rotor * f_idle
    # (2.5) hover command under the TRUE actuator map
    om_hover = float(np.sqrt(m * P.g / (P.n_rotor * P.K_T)))
    u_hover = float(cmd_of_omega(om_hover))
    a_lat_max = float(np.sqrt((T_max / m) ** 2 - P.g ** 2))               # (2.6)
    # (2.11) hover control effectiveness, d a_z/d c = 2 n K_T Om_h (Om_max-Om_min)/m
    daz_dc = float(2 * P.n_rotor * P.K_T * om_hover * (P.Om_max - P.Om_min) / m)
    arms = R_ROTOR - cg_nom
    M_nom = _alloc(arms, P.k_m, SIGMA)                                    # (2.7) truth
    M_ctrl = _alloc(arms, P.k_m_ctrl, SIGMA)          # what the PX4 allocator believes
    return dict(m=m, cg_nom=cg_nom, J_nom=J_nom, Jinv_nom=np.linalg.inv(J_nom),
                f_max=f_max, T_max=T_max, f_idle=f_idle, T_idle=T_idle,
                TW=T_max / (m * P.g), u_hover=u_hover, om_hover=om_hover,
                daz_dc=daz_dc, a_lat_max=a_lat_max,
                M_nom=M_nom, Minv_nom=np.linalg.inv(M_nom),
                M_ctrl=M_ctrl, Minv_ctrl=np.linalg.inv(M_ctrl),
                Mtau_nom=M_nom[1:], arms_nom=arms)


_D = _derive()
M_TOT, CG_NOM, J_NOM, JINV_NOM = _D["m"], _D["cg_nom"], _D["J_nom"], _D["Jinv_nom"]
F_MAX, T_MAX, TW, U_HOVER = _D["f_max"], _D["T_max"], _D["TW"], _D["u_hover"]
T_IDLE, OM_HOVER = _D["T_idle"], _D["om_hover"]
A_LAT_MAX = _D["a_lat_max"]
M_NOM, MINV_NOM, MTAU_NOM, ARMS_NOM = _D["M_nom"], _D["Minv_nom"], _D["Mtau_nom"], _D["arms_nom"]
#: the allocator's belief (k_m = 0.05), distinct from the plant's truth (A3)
M_CTRL, MINV_CTRL = _D["M_ctrl"], _D["Minv_ctrl"]

# The J off-diagonals must vanish by symmetry (§2.2); assert it at import so a
# bad rotor table cannot pass silently.
assert np.abs(J_NOM - np.diag(np.diag(J_NOM))).max() < 1e-15, "J off-diagonals nonzero"
assert abs(float(np.linalg.cond(M_NOM)) - 62.5) < 1e-6, "cond(M) != 62.5"

OM_MAX = np.asarray(P.om_max)
#: (2.11) hover control effectiveness under the TRUE actuator map (A2).
#: = 2 n K_T Om_hover (Om_max - Om_min) / m = 21.6670 s^-2.
#: The pure-square form 2 T_max u_hover / m = 25.4905 overstates it by 17.65 %.
DAZ_DC_HOVER = _D["daz_dc"]


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
        # the PX4 allocator inverts with k_m_ctrl = 0.05 while the rotors
        # produce k_m = 0.016: a real 3.125x yaw-authority error (A3)
        Minv_ctrl=jnp.broadcast_to(jnp.asarray(MINV_CTRL), (B, 4, 4)),
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
    """(2.10) velocity-type motor model, with the PX4 idle floor (A2).

        T(c) = n K_T (Om_min + c (Om_max - Om_min))^2

    ``T_max`` is the thrust at c = 1 and is carried per-vehicle so that the
    lambda_T plant perturbation scales it; the idle floor scales with it.
    Thrust is still quadratic in rotor speed -- the motor is velocity-type --
    but it is **not** proportional to c^2, because c = 0 is 150 rad/s and not 0.
    """
    r = P.Om_min / P.Om_max
    return T_max * (r + _clamp(c, 0.0, 1.0) * (1.0 - r)) ** 2


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
    # (3.2) with the SAME 220 deg/s clip the autopilot applies (D7).  om_max is
    # (10,10,4) rad/s but rate_cmd saturates at 3.84, so without this clip the
    # optimiser plans in a region 62 % of whose roll/pitch box the plant cannot
    # reach, and model and plant disagree above 3.84 rad/s.
    om = _clamp(jnp.asarray(OM_MAX) * u[..., 1:4], -P.rate_max, P.rate_max)
    a_res = jnp.zeros_like(v) if d is None else d[..., 0:3]
    om_res = jnp.zeros_like(om) if d is None else d[..., 3:6]
    acc = thrust_of(u[..., 0:1], T_max) / m * qzaxis(q) - P.g * jnp.asarray(E3) + a_res
    qdot = 0.5 * qmul(q, jnp.concatenate(
        [jnp.zeros_like(om[..., :1]), om + om_res], -1))
    return jnp.concatenate([v, acc, qdot], -1)


@functools.partial(jax.jit, static_argnames=("dt",))
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


@jax.jit
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
    # the idle floor is a floor on the ACTUATOR, so the allocator cannot ask for
    # less than it (A2)
    Om_cmd = _clamp(Om_cmd, P.Om_min, P.Om_max)
    # PWM normalised on [Om_min, Om_max] -- the scaling /fmu/out/actuator_motors
    # actually publishes.  Om_cmd/Om_max would differ in both slope and offset
    # and would put the RDP's 4 PWM channels on a different scale in deployment
    # from the one they were trained on (A4).
    return Om_cmd, cmd_of_omega(Om_cmd), tau_cmd


@jax.jit
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
    # SDF rotorDragCoefficient / rollingMomentCoefficient (A5).  Both act on the
    # airspeed component PERPENDICULAR to the rotor axis and scale with rotor
    # speed; they were previously absent, the bulk D standing in for all of it.
    zb = qzaxis(q)
    v_air = v - v_w
    v_perp = v_air - zb * jnp.sum(v_air * zb, -1, keepdims=True)
    sum_om = jnp.sum(Om, -1, keepdims=True)
    F_rd = -P.c_rotor_drag * sum_om * v_perp
    v_perp_b = jnp.einsum("bji,bj->bi", R, v_perp)
    tau_rm = -P.c_roll_mom * sum_om * v_perp_b
    acc = (thrust / par["m"] * zb - P.g * jnp.asarray(E3)                   # (3.6)
           - par["D"] * v_air / par["m"]
           + F_rd / par["m"]
           + jnp.einsum("bij,bj->bi", R, F_b) / par["m"])
    tau = (jnp.einsum("bij,bj->bi", par["Mtau_w2"], w2) + tau_b + tau_rm    # (3.7)
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


@functools.partial(jax.jit, static_argnames=("n_sub",))
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
    f_h = par["m"] * P.g / P.n_rotor
    Om = jnp.broadcast_to(jnp.sqrt(f_h / par["KT"]), (B, 4))
    z = jnp.zeros((B, 3))
    return jnp.concatenate([p, q, z, z, Om, z, z], -1)


def hover_u(B=1, par=None):
    """The CTBR command that trims :func:`hover_state`."""
    par = make_par(B) if par is None else par
    # invert (2.10) with the idle floor: c = (sqrt(T/T_max) - r)/(1 - r) (A2)
    r = P.Om_min / P.Om_max
    c = (jnp.sqrt(par["m"] * P.g / par["T_max_ctrl"]) - r) / (1.0 - r)
    return jnp.concatenate([c, jnp.zeros((B, 3))], -1)


# --------------------------------------------------------------------------- #
# §5.5  error coordinates
# --------------------------------------------------------------------------- #
@jax.jit
def err(x, xr):
    """(4.1) reduced error, (B,10),(B,10) -> (B,9).

    The sign of q_e picks the shorter rotation, so ||e[6:9]|| = 2 sin(theta/2).
    """
    qe = qmul(qconj(xr[..., 6:10]), x[..., 6:10])
    sgn = jnp.where(qe[..., :1] >= 0.0, 1.0, -1.0)
    return jnp.concatenate([x[..., 0:3] - xr[..., 0:3],
                            x[..., 3:6] - xr[..., 3:6], 2.0 * sgn * qe[..., 1:4]], -1)


@jax.jit
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
        # exact, and now consistent with _ref_pva for every omega including a
        # capped one (D5): |v| = hypot(R w, HELIX_CLIMB * R w) = R w * kappa_v
        v_h = R * omega
        return float(np.hypot(v_h, HELIX_CLIMB * v_h)), float(R * omega ** 2 * KAPPA_A[i])
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
        elif i == 4:
            # Helix climb (D5).  The rate follows the REALISED horizontal speed
            # R*omega, not the sampled spd: when (4.6)'s feasibility cap binds,
            # R*omega != spd and using spd makes the climb inconsistent with the
            # circle it is wrapped around -- and breaks the dagger condition
            # that KAPPA_V[helix] is stated under.
            v_h = R * w
            climb = HELIX_CLIMB * v_h * t[:, None]
            p = p + climb * jnp.asarray(E3)
            v = v + HELIX_CLIMB * v_h * jnp.asarray(E3)
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


@functools.partial(jax.jit, static_argnames=("hold",))
def ref_state(ep, t, hold=False):
    """(B,10) reference state, (B,4) u_ref (4.8), (B,3) a_ref.

    omega_ref comes from a centred difference of q_ref at +/- dt_c/2 (§5.6).
    The collective in (4.8) carries a **square root** -- it is exact, because
    thrust is quadratic in the command (2.10).
    """
    t = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    p, v, a = _ref_pva(ep, t)
    if hold:
        # position hold: keep the point, drop the motion.  v = a = 0 makes
        # q_ref the identity and u_ref the hover command, which is what a
        # regulation task actually asks for.
        v = jnp.zeros_like(v)
        a = jnp.zeros_like(a)
    q = ref_attitude(a)
    h = 0.5 * P.dt_c
    if hold:
        qdot = jnp.zeros_like(q)
    else:
        qp = ref_attitude(_ref_pva(ep, t + h)[2])
        qm = ref_attitude(_ref_pva(ep, t - h)[2])
        qdot = (qp - qm) / (2 * h)
    om_ref = 2.0 * qmul(qconj(q), qdot)[..., 1:4]              # body rates
    # (4.8): exact inversion of (2.10) INCLUDING the idle floor (A2).  The bare
    # square root would be 5.6 % high at hover on the real platform.
    r = P.Om_min / P.Om_max
    root = jnp.sqrt(jnp.clip(
        M_TOT * jnp.linalg.norm(a + P.g * jnp.asarray(E3), axis=-1, keepdims=True)
        / T_MAX, 0.0, 1.0))
    c = jnp.clip((root - r) / (1.0 - r), 0.0, 1.0)
    u_ref = jnp.concatenate([c, om_ref / jnp.asarray(OM_MAX)], -1)
    return jnp.concatenate([p, v, q], -1), u_ref, a


# --------------------------------------------------------------------------- #
# §5.7  disturbance accessors -- (a) model residual vs (b) physical wrench
# --------------------------------------------------------------------------- #
@jax.jit
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


@jax.jit
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


def moment_gain(t):
    """Exact closed-loop gain of the rate loop to a STEP external moment.

    Neglecting K_d and the gyro filter, the loop of (3.4) gives

        eps_ddot + K_r eps_dot + K_i eps = 0 ,   eps(0) = 0 , eps_dot(0) = tau/J

    so  eps(t) = (tau/J)(e^{r+ t} - e^{r- t})/(r+ - r-).  This returns
    ``eps(t) / (tau/(J K_r))``, i.e. the factor by which the true rate residual
    differs from (4.13), which is the K_i = 0 steady state and therefore the
    t -> small limit (gain 1).

    Measured against the full plant: 0.87 at t = 0.1 s, 0.93 at 0.3 s, 0.43 at
    3 s, **0.103 at 8 s**.  (4.13) is a SHORT-TRANSIENT model.  Applied to a
    standing moment that has been acting for seconds -- which is exactly the
    `asym` payload scenario -- it overstates the rate residual by ~9x (D3/D4).
    """
    Kr, Ki = np.asarray(P.K_rate), np.asarray(P.K_i)
    disc = np.sqrt(np.maximum(Kr ** 2 - 4 * Ki, 1e-300))
    rp, rm = (-Kr + disc) / 2, (-Kr - disc) / 2
    return Kr * (np.exp(rp * t) - np.exp(rm * t)) / (rp - rm)


#: Modes for :func:`wrench_to_dmod`.  ``first_order`` is (4.13) verbatim;
#: ``none`` zeroes the moment block, which is the CORRECT limit for a moment the
#: rate integrator has already absorbed; ``closed_loop`` scales (4.13) by
#: :func:`moment_gain` at ``DMOD_SETTLE_S``.
DMOD_MODES = ("first_order", "none", "closed_loop")
#: Time a disturbance is assumed to have been acting when ``closed_loop`` is
#: used.  0.5 s is ~2 MPC horizons at N = 10 and well inside the integral time
#: constant, so it brackets the transient the MPC can actually act on.
DMOD_SETTLE_S = 0.5


@functools.partial(jax.jit, static_argnames=("mode",))
def wrench_to_dmod(w, mode: str = "first_order"):
    """(4.12)+(4.13) convert a wrench [N, N m] to a model residual [m/s^2, rad/s].

    Force converts exactly, a_res = F/m.  The moment block has **no exact
    image** -- the 10-state model has no torque input -- so (4.13) uses the
    steady state of the rate loop with K_i neglected.

    **Validity (D3).**  That neglect is the whole content of the approximation.
    :func:`moment_gain` gives the exact factor: ~0.9 out to 0.3 s, 0.10 by 8 s.
    For a genuine transient ``first_order`` is right; for a standing moment the
    integrator has absorbed it and the true rate residual is near zero, which is
    what ``none`` encodes.  ``closed_loop`` interpolates.  Report both, as §5.7
    requires.
    """
    a_res = w[..., 0:3] / M_TOT
    if mode == "none":
        om_res = jnp.zeros_like(a_res)
    else:
        G = jnp.asarray(np.linalg.inv(np.diag(P.K_rate) @ J_NOM))
        om_res = jnp.einsum("ij,bj->bi", G, w[..., 3:6])              # (4.13)
        if mode == "closed_loop":
            om_res = om_res * jnp.asarray(moment_gain(DMOD_SETTLE_S))
    return jnp.concatenate([a_res, om_res], -1)


# --------------------------------------------------------------------------- #
# §5.8  the differentiable MPC layer
# --------------------------------------------------------------------------- #
#: CTBR input box.  The rate entries are the **reachable** set, not the nominal
#: [-1,1] (D7): om_max is (10,10,4) rad/s but the autopilot clips the rate
#: command at rate_max = 3.84 rad/s, so |ubar| beyond rate_max/om_max = 0.384 on
#: roll and pitch commands a rate the plant cannot produce.  Constraining the
#: box to the reachable set makes the control model and the plant agree
#: everywhere *inside* it, which a clip inside fc() cannot do: a clip leaves the
#: derivative zero beyond the knee, so Q_uu goes singular in the rate directions
#: exactly when the solver tries to come back, and the collective absorbs the
#: difference.  Measured, that raised untrained saturation to 47 %.
_U_RATE = np.minimum(1.0, P.rate_max / np.asarray(P.om_max))
U_LO = np.concatenate([[0.0], -_U_RATE])
U_HI = np.concatenate([[1.0], _U_RATE])
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
    r = P.Om_min / P.Om_max
    root = jnp.sqrt(jnp.clip(M_TOT * jnp.linalg.norm(zb, axis=-1, keepdims=True) / T_MAX,
                             0.0, 1.0))
    c = jnp.clip((root - r) / (1.0 - r), 0.0, 1.0)
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
#: Re-derived after the physics corrections (A2: control effectiveness 21.67
#: rather than 25.49; D7: the rate box is the reachable set).  Over the grid
#: RMSE now runs 0.0506 m to 1.0552 m, a best-to-worst ratio of 20.8.
Q_HAND = np.diag([10.0, 10.0, 10.0, 4.0, 4.0, 4.0, 1.0, 1.0, 0.5])
R_HAND = np.diag([0.5, 5.0, 5.0, 5.0])

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
#: Range of a learned stage weight.  Five orders of magnitude, mapped
#: **logarithmically**: S_ii = Q_LO * (Q_HI/Q_LO)^sigmoid(z), so sigmoid(z)=0.5
#: lands on the geometric mean sqrt(Q_LO*Q_HI) = 3.16 rather than the arithmetic
#: one.  See C-7: a linear map over five decades puts a zero-initialised head at
#: ~5e4, which is 660x the terminal matrix, and at N=1 the only free variable is
#: du -- so the solver returns |du| = 1.9e-5, the controller degenerates to pure
#: feed-forward, and the cost map has no effect on anything.  A linear map also
#: wastes almost all of the policy's resolution: 90 % of its output range would
#: sit inside the top decade.
Q_LO, Q_HI = 1.0e-2, 1.0e3
#: Bound on the learned **linear** cost term p of (5.2).  arXiv:2306.09852 §III-G
#: learns p as well as Q -- its cost-map output dimension is 2T(n_state+n_input)
#: -- and bounds both with the same sigmoid.  A strictly positive p is not
#: meaningful in *error* coordinates, where the target is e = 0: it would bias
#: every channel one way.  p is therefore bounded symmetrically, p = P_HI*tanh(z),
#: with P_HI kept BELOW the geometric-mean quadratic weight sqrt(Q_LO*Q_HI) =
#: 3.16 so the linear term can shift the optimum without dominating it: the
#: unconstrained minimiser of 1/2 S t^2 + p t sits at t = -p/S, so P_HI/Q_geo is
#: directly the largest offset the linear term alone can command (B2).
P_HI = 2.0
REPS = ("diag", "chol", "full")
#: Q block + p block, matching the paper's 2T(n+m) for the diagonal case.
REP_DIM = {"diag": 2 * NTAU, "chol": NTAU * (NTAU + 1) // 2 + NTAU,
           "full": NTAU * NTAU + NTAU}
REP_Q_DIM = {"diag": NTAU, "chol": NTAU * (NTAU + 1) // 2, "full": NTAU * NTAU}
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


def normalise_obs(theta, obs):
    """Apply the actor's stored running observation statistics (B4).

    arXiv:2306.09852 §IV-A normalises the observation with the running mean and
    standard deviation, recomputed each training iteration.  Without it the
    trunk sees a vector whose blocks differ by orders of magnitude -- the oracle
    channel alone is a wrench in N and N.m, ~2.3x the rms of everything else --
    and the first layer has to undo that before it can learn anything.
    The statistics travel WITH the checkpoint, so inference normalises exactly
    as training did.
    """
    nrm = theta.get("obs_norm")
    if nrm is None:
        return obs
    # Floor the variance and clip the result, as every production PPO
    # normaliser does.  Several observation channels are near-constant within a
    # batch -- a frozen preview under position hold, an integral that has not
    # moved -- and an unfloored 1/sqrt(var) then amplifies the first sample that
    # does move by ~1e4, which takes the policy to NaN within three iterations.
    return jnp.clip((obs - nrm["mu"]) / jnp.sqrt(jnp.maximum(nrm["var"], OBS_VAR_FLOOR)),
                    -OBS_CLIP, OBS_CLIP)


def costmap_head(theta, obs, rep, N):
    """Trunk + head -> the raw per-stage parameter block, (B,N,REP_DIM[rep])."""
    h = mlp_apply(theta["trunk"], normalise_obs(theta, obs))
    z = mlp_apply(theta["head"], h)
    return z.reshape(obs.shape[0], N, REP_DIM[rep])


#: Variance floor and clip for :func:`normalise_obs`.
OBS_VAR_FLOOR, OBS_CLIP = 1e-4, 10.0


def obs_norm_init(obs_dim):
    return {"mu": jnp.zeros((obs_dim,)), "var": jnp.ones((obs_dim,)),
            "count": jnp.asarray(1e-4)}


def obs_norm_update(nrm, batch):
    """Welford-style running update over a flat (n, obs_dim) batch."""
    bm, bv, bn = batch.mean(0), batch.var(0), float(batch.shape[0])
    d = bm - nrm["mu"]
    tot = nrm["count"] + bn
    mu = nrm["mu"] + d * bn / tot
    m_a = nrm["var"] * nrm["count"]
    m_b = bv * bn
    var = (m_a + m_b + d ** 2 * nrm["count"] * bn / tot) / tot
    return {"mu": mu, "var": var, "count": tot}


def costmap_from_z(z, rep):
    """Raw head output (B,N,REP_DIM) -> (S (B,N,13,13) PSD, c (B,N,13)).

    The **single** implementation, shared by training and by inference.  Keeping
    two copies of this map in step is exactly the kind of silent divergence §11
    exists to catch: the policy would be optimising one cost and the deployed
    controller solving another, with no error anywhere.

    The trailing NTAU entries are the linear term p (B2); everything before them
    parameterises S.
    """
    B, N = z.shape[0], z.shape[1]
    nq = REP_Q_DIM[rep]
    zq, zp = z[..., :nq], z[..., nq:]
    if rep == "diag":
        dg = Q_LO * (Q_HI / Q_LO) ** jax.nn.sigmoid(zq)         # log-spaced
        S = jnp.einsum("bnij,bnj->bnij",
                       jnp.broadcast_to(jnp.eye(NTAU), (B, N, NTAU, NTAU)), dg)
    elif rep == "chol":
        A = jnp.zeros((B, N, NTAU, NTAU)).at[..., _TRIL[0], _TRIL[1]].set(zq)
        di = jnp.arange(NTAU)
        A = A.at[..., di, di].set(jax.nn.softplus(A[..., di, di]) + 1e-6)
        S = jnp.einsum("bnij,bnkj->bnik", A, A) + Q_LO * jnp.eye(NTAU)
    elif rep == "full":
        A = zq.reshape(B, N, NTAU, NTAU)
        S = jnp.einsum("bnij,bnkj->bnik", A, A) + Q_LO * jnp.eye(NTAU)
    else:
        raise ValueError(rep)
    return S, P_HI * jnp.tanh(zp)


def costmap_apply(theta, obs, rep, N):
    """-> S (B,N,13,13) PSD, c (B,N,13).

    Initialisation asymmetry, recorded because it matters (§5.9): with a
    0.1-scaled zero-bias head, ``chol``/``full`` start at A ~ 0 so S ~ Q_LO*I --
    the quadratic term is effectively *absent* at init -- whereas ``diag``
    starts at the sigmoid mid-point, which under the log map of C-7 is the
    geometric mean sqrt(Q_LO*Q_HI) = 3.16.  The three do not start from
    comparable places, which predicts the richer forms need **longer**, not that
    they are incapable.
    """
    return costmap_from_z(costmap_head(theta, obs, rep, N), rep)


def costmap_init(key, obs_dim, hid, rep, N, n_layer=2, normalise=True):
    """Cost map: trunk -> head producing the Q and p blocks for all N stages."""
    k1, k2 = jax.random.split(key, 2)
    th = {"trunk": mlp_init(k1, [obs_dim] + [hid] * n_layer,
                            scale_last=np.sqrt(2.0 / hid)),
          "head": mlp_init(k2, [hid, N * REP_DIM[rep]], scale_last=0.1)}
    if normalise:
        th["obs_norm"] = obs_norm_init(obs_dim)
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


def build_obs(e, xr, prev, ep, t, s, oracle=None, hold=False):
    """(5.3), OBS_DIM = 40, plus 6 when the oracle channel is on.

    Preview stride is **5 control steps**, i.e. lookaheads 0.1/0.2/0.3 s; it is
    otherwise invisible.  ``prev`` carries the integral and the running means.
    """
    p = s[..., SP]
    prev_blocks = []
    for h in PREVIEW_H:
        # under position hold the preview is the same fixed setpoint at every
        # lookahead, which is exactly the information a hold task has
        t_h = t if hold else t + h * PREVIEW_STRIDE * P.dt_c
        pr, vr, _ = _ref_pva(ep, t_h)
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
    #: 'track' follows the sampled path; 'stabilize' is POSITION-HOLD -- the
    #: reference is frozen at its t=0 point, so the vehicle regulates to a fixed
    #: setpoint.  arXiv:2605.16015 trains the adaptive policy this way and argues
    #: it is what produces aggressive recovery that still generalises to
    #: tracking.  Previously this field existed and was read nowhere (E3).
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


@jax.jit
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
        """Sample an episode.

        ``fixed`` is applied **twice**: once to the primitives (kind, R, spd,
        ...) *before* omega is derived from them through (4.6), and once at the
        end so that omega itself, or any derived field, can still be pinned
        explicitly.  Applying it only at the end -- the obvious way -- leaves a
        pinned radius paired with the omega of the radius that was sampled, so
        a controlled experiment silently varies the thing it pinned.
        """
        ks = jax.random.split(key, 20)
        pidx = jnp.asarray([PATH_IDX[p] for p in self.cfg.paths])
        az = _u(ks[7], (0.0, 2 * np.pi), (n,))
        ep = dict(
            kind=pidx[jax.random.randint(ks[0], (n,), 0, len(pidx))],
            c=jnp.stack([_u(ks[1], (-1.5, 1.5), (n,)), _u(ks[2], (-1.5, 1.5), (n,)),
                         _u(ks[3], (1.0, 2.5), (n,))], -1),
            R=_u(ks[4], (0.5, 2.0), (n,)),
            phi0=_u(ks[5], (0.0, 2 * np.pi), (n,)),
            spd=_u(ks[6], self.dist["speed"], (n,)),
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
        ep = self._apply_fixed(ep)                 # primitives, before (4.6)
        ep["omega"] = jnp.take_along_axis(
            jnp.stack([path_omega(i, ep["R"], ep["spd"])
                       for i in range(len(PATHS_ALL))], -1), ep["kind"][:, None], 1)[:, 0]
        return self._apply_fixed(ep)               # derived fields, incl. omega

    def _apply_fixed(self, ep):
        for k, v in self.fixed.items():
            if k in ep:
                ep[k] = jnp.broadcast_to(jnp.asarray(v, dtype=jnp.float64),
                                         jnp.shape(ep[k]))
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
        xr, _, _ = ref_state(ep, jnp.zeros(n), hold=self.cfg.task == "stabilize")
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
    def _t_ref(self, t=None):
        """Reference clock.  Under ``task='stabilize'`` it is pinned at 0, which
        freezes the reference at its initial point: position hold (E3)."""
        t = self.t if t is None else t
        return jnp.zeros_like(t) if self.cfg.task == "stabilize" else t

    def ref_now(self):
        return ref_state(self.ep, self._t_ref(), hold=self.cfg.task == "stabilize")

    def ref_traj(self, N):
        """(B,N+1,10): the preview that (5.1) scores against."""
        if self.cfg.task == "stabilize":
            xr = ref_state(self.ep, self._t_ref(), hold=True)[0]
            return jnp.broadcast_to(xr[:, None], (self.n, N + 1, NX))
        return _ref_traj(self.ep, self.t, N)

    def ref_useq(self, N):
        """(B,N,4): the **exact analytic** u_ref over the horizon (N-3)."""
        if self.cfg.task == "stabilize":
            ur = ref_state(self.ep, self._t_ref(), hold=True)[1]
            return jnp.broadcast_to(ur[:, None], (self.n, N, NU))
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
        """The extra 6 observation channels, when they are switched on.

        Routed through :meth:`d_channel` / :meth:`dmod`, **not** through
        :meth:`d_truth`: with no estimator attached these are the truth, which is
        what makes the arm an oracle, but once an RDP is attached they become the
        *prediction*, which is what makes variants A and C use the estimator they
        claim to.  Reading ``d_truth`` here would hand ground truth to the online
        controller -- §9.1 exposes it to the logger and the evaluator only -- and
        the RDP rows would silently be oracle rows.
        """
        if not self.cfg.oracle:
            return None
        return (self.d_channel() if self.cfg.oracle_target == "wrench"
                else self.dmod())

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
    hold = cfg.task == "stabilize"
    t_ref = jnp.zeros_like(t) if hold else t
    xr, _, _ = ref_state(ep, t_ref, hold=hold)
    e = err(plant_to_ctrl(state), xr)
    lv = NOISE_LEVELS[cfg.noise]
    if any(v > 0 for v in lv.values()):
        ks = jax.random.split(key, 3)
        n = state.shape[0]
        e = e + jnp.concatenate([lv["p"] * jax.random.normal(ks[0], (n, 3)),
                                 lv["v"] * jax.random.normal(ks[1], (n, 3)),
                                 lv["q"] * jax.random.normal(ks[2], (n, 3))], -1)
    return build_obs(e, xr, prev, ep, t_ref, state, hold=hold), e, xr


@functools.partial(jax.jit, static_argnums=(0, 1))
def _step_jit(cfg, no_respawn, state, par, ep, prev, t, n_step, u,
              s_new, par_new, ep_new, prev_new):
    u = jnp.clip(u, jnp.asarray(U_LO), jnp.asarray(U_HI))
    hold = cfg.task == "stabilize"
    t_ref = jnp.zeros_like(t) if hold else t
    xr, uref, _ = ref_state(ep, t_ref, hold=hold)
    e = err(plant_to_ctrl(state), xr)
    d_u, du = u - prev["u"], u - uref
    s_next = step_p(state, u, par, wind_at(ep, t))
    t_next = t + P.dt_c
    xr_n, _, _ = ref_state(ep, jnp.zeros_like(t_next) if hold else t_next, hold=hold)
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
                    max_grad_norm=0.5, lr=3e-4, epochs=10, minib=32, sigma=0.05,
                    algo="ppo", mpve=False, mpve_coef=0.5, kl_target=0.01,
                    lr_decay=0.5, lr_grow=1.2, lr_min=1e-7, rep="diag", N=1,
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


def mpve_value_loss(critic, obs_n, e_seq, du_seq, om, d_u, gamma):
    """(8)-(9) Model-Predictive Value Expansion.

    The differentiable MPC already produced a predicted trajectory; MPVE prices
    it instead of discarding it.  (8) gives the H-step target

        V_H(s) = sum_t gamma^t r_hat_t + gamma^H V(s_H)

    and (9) extends the value loss with the TD-k consistency term over every
    intermediate prediction, which is what aligns the training distribution with
    the prediction distribution.

    **Documented approximation.**  The critic is V(o) over the 40-D observation,
    while the MPC predicts the 9-D error only.  A predicted observation is
    therefore built by substituting the predicted error into the first 9 entries
    and holding the preview/integral/history blocks at their current values.
    Over a 20-200 ms horizon those move very little, but this is an
    approximation, not an identity.
    """
    B, N = obs_n.shape[0], du_seq.shape[1]
    sub = lambda e: jnp.concatenate([e, obs_n[:, NE:]], -1)
    zc = jnp.zeros(B)
    r_hat = jnp.stack([reward_quad(e_seq[:, k + 1], du_seq[:, k], om, d_u, zc)
                       for k in range(N)], 1)
    V_pred = jnp.stack([mlp_apply(critic, sub(e_seq[:, k]))[:, 0]
                        for k in range(N + 1)], 1)
    loss = 0.0
    for t in range(N):                                               # eq (9)
        tgt = sum((gamma ** (k - t)) * r_hat[:, k] for k in range(t, N))
        tgt = tgt + (gamma ** (N - t)) * V_pred[:, N]
        loss = loss + jnp.mean((V_pred[:, t] - jax.lax.stop_gradient(tgt)) ** 2)
    return loss / max(N, 1)


def critic_init(key, obs_dim, hid, n_layer=2):
    return mlp_init(key, [obs_dim] + [hid] * n_layer + [1], scale_last=0.01)


def _logp(a, mu, log_sigma):
    z = (a - mu) / jnp.exp(log_sigma)
    return jnp.sum(-0.5 * z ** 2 - log_sigma - 0.5 * np.log(2 * np.pi), -1)


def train_ppo(env_fn, cfg, seed=0, actor=None, critic=None, log_path=None,
              iters=100, T_rollout=64, verbose=True, env=None):
    """PPO (Schulman 2017) over the differentiable MPC layer, per arXiv:2306.09852.

    **The policy is a Gaussian over the ACTION**, exactly as the paper states:

        u ~ N{ diffMPC(x_k, Q(s_k), p(s_k)), Sigma }                     (7)

    The cost map is a deterministic function of the observation; the stochastic
    part is the command.  An earlier version perturbed the cost-map *parameters*
    instead, which changes the policy's support entirely: measured, saturation
    then does not rise with sigma at all (6.2 -> 6.2 -> 4.7 % for sigma 0.05 ->
    0.30), whereas perturbing the action drives it monotonically (10.9 -> 15.6
    -> 29.7 %).  §8.3's exploration study is *about* that mechanism, so it
    cannot be run on parameter noise (B1).

    Consequence, and it is the paper's too: the MPC is **re-solved on every
    minibatch of every epoch**, because the policy mean depends on theta through
    the solve -- "for every backward and forward pass of the actor network, we
    need to solve an optimization problem."

    ``sat`` is a **gate, not a diagnostic** (§5.13): if collective saturation
    exceeds ``cfg['sat_gate']`` over the last 20 % of iterations a loud warning
    is printed naming the disturbance magnitude.  A policy pinned against its
    input box never learns a disturbance-response manifold, and every downstream
    comparison built on it is void.
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
    N, rep, use_d = c["N"], c["rep"], bool(c.get("use_d", False))
    # B5: the exploration std is a LEARNED parameter, as in PPO proper.  Held
    # fixed it can never anneal, and with ent_coef = 0 nothing else moves it.
    params = {"actor": actor, "critic": critic,
              "log_sigma": jnp.full((NU,), np.log(c["sigma"]))}
    # Adaptive step size, keyed to the measured KL (B1 consequence).  The MPC
    # mean is orders of magnitude more sensitive to its cost map than an MLP's
    # output is to its weights, and that sensitivity is problem- and
    # scale-dependent, so a fixed lr either crawls or leaves the trust region on
    # its first step.  lr is shrunk when the region is breached and grown when
    # there is slack, which is the PPO analogue of what TRPO does exactly.
    opt = optax.inject_hyperparams(
        lambda lr: optax.chain(optax.clip_by_global_norm(c["max_grad_norm"]),
                               optax.adam(lr)))(lr=c["lr"])
    opt_state = opt.init(params)
    lr_cur = float(c["lr"])

    @jax.jit
    def mpc_mean(actor_p, o, e, xr_seq, uref_seq, d):
        """The policy mean: the differentiable MPC's first command."""
        S, cc = costmap_apply(actor_p, o, rep, N)
        Pt = jnp.broadcast_to(PTt, (o.shape[0], NE, NE))
        du = ilqr_solve(e, xr_seq, S, cc, Pt, d, None, c["n_iter"],
                        c["n_diff"], uref_seq)
        return uref_seq[:, 0] + du[:, 0], du

    @jax.jit
    def rollout_mean(actor_p, o, e, xr_seq, uref_seq, d):
        return mpc_mean(actor_p, o, e, xr_seq, uref_seq, d)[0]

    rows = []
    for it in range(iters):
        keys = ("o", "e", "xr", "ur", "d", "a", "lp", "rw", "vl", "dn")
        BUF = {k: [] for k in keys}
        sat_acc, crash_acc = [], []
        o, e, xr = env.obs()
        for t in range(T_rollout):
            key, k = jax.random.split(key)
            xr_seq, uref_seq = env.ref_traj(N), env.ref_useq(N)
            d = env.dmod() if use_d else jnp.zeros((o.shape[0], 6))
            mu = rollout_mean(params["actor"], o, e, xr_seq, uref_seq,
                              d if use_d else None)
            a = mu + jnp.exp(params["log_sigma"]) * jax.random.normal(k, mu.shape)
            lp = _logp(a, mu, params["log_sigma"])                       # (7)
            v = mlp_apply(params["critic"], normalise_obs(params["actor"], o))[:, 0]
            r, done, info = env.step(a)          # the env clips into the box
            for kk, vv in zip(keys, (o, e, xr_seq, uref_seq, d, a, lp, r, v,
                                     done.astype(jnp.float64))):
                BUF[kk].append(vv)
            sat_acc.append(float(info["sat"]))
            crash_acc.append(float(jnp.mean(info["crash"])))
            o, e, xr = env.obs()
        v_last = mlp_apply(params["critic"], normalise_obs(params["actor"], o))[:, 0]
        ST = {k: jnp.stack(v) for k, v in BUF.items()}
        ADV = gae(ST["rw"], ST["vl"], v_last, ST["dn"], c["gamma"], c["lam"])
        RET = ADV + ST["vl"]
        flat = lambda z: z.reshape((-1,) + z.shape[2:])
        F = {k: flat(v) for k, v in ST.items()}
        ADVf, RETf = flat(ADV), flat(RET)
        ADVf = (ADVf - ADVf.mean()) / (ADVf.std() + 1e-8)

        def loss_fn(p, ob, ee, xrs, urs, dd, aa, lp_old, adv, ret):
            mu, du = mpc_mean(p["actor"], ob, ee, xrs, urs, dd if use_d else None)
            lp = _logp(aa, mu, p["log_sigma"])
            # guard the ratio: a diverged actor can otherwise produce inf here
            # and take the whole update to NaN before the KL trip can fire
            ratio = jnp.exp(jnp.clip(lp - lp_old, -20.0, 20.0))
            l_pi = -jnp.mean(jnp.minimum(
                ratio * adv, jnp.clip(ratio, 1 - c["clip"], 1 + c["clip"]) * adv))
            von = normalise_obs(p["actor"], ob)
            l_v = jnp.mean((mlp_apply(p["critic"], von)[:, 0] - ret) ** 2)
            if c["mpve"]:                                    # B3: actually used
                e_seq = rollout_err(ee, du, xrs, urs, dd if use_d else None)
                zb3 = jnp.zeros((ob.shape[0], 3))
                l_v = l_v + c["mpve_coef"] * mpve_value_loss(
                    p["critic"], von, e_seq, du, zb3,
                    jnp.zeros((ob.shape[0], NU)), c["gamma"])
            ent = jnp.sum(p["log_sigma"] + 0.5 * np.log(2 * np.pi * np.e))
            loss = l_pi + c["vf_coef"] * l_v - c["ent_coef"] * ent
            kl = jnp.mean(lp_old - lp)
            cf = jnp.mean((jnp.abs(ratio - 1.0) > c["clip"]).astype(jnp.float64))
            return loss, (l_pi, l_v, ent, kl, cf)

        grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        n = F["o"].shape[0]
        mb = max(n // c["minib"], 1)
        aux_last, gnorm, kl_seen = None, 0.0, 0.0
        args_all = (F["o"], F["e"], F["xr"], F["ur"], F["d"], F["a"], F["lp"],
                    ADVf, RETf)
        if c["algo"] == "trpo":
            params["actor"], kl_a, impr = trpo_step(
                params["actor"], params["log_sigma"],
                lambda ap, ob: mpc_mean(ap, ob, F["e"], F["xr"], F["ur"],
                                        F["d"] if use_d else None)[0],
                F["o"], F["a"], F["lp"], ADVf, max_kl=c["kl_target"])
            for _ in range(c["epochs"]):
                (loss, aux_last), g = grad_fn(params, *args_all)
                gv = jax.tree_util.tree_map(jnp.zeros_like, g)
                gv["critic"] = g["critic"]
                upd, opt_state = opt.update(gv, opt_state, params)
                params = optax.apply_updates(params, upd)
                gnorm = float(optax.global_norm(g["critic"]))
            aux_last = (aux_last[0], aux_last[1], aux_last[2],
                        jnp.asarray(kl_a), aux_last[4])
        else:
            stop = False
            for _ in range(c["epochs"]):
                if stop:
                    break
                key, k = jax.random.split(key)
                perm = jax.random.permutation(k, n)
                for i in range(c["minib"]):
                    idx = perm[i * mb:(i + 1) * mb]
                    if idx.size == 0:
                        continue
                    mbargs = [z[idx] for z in args_all]
                    (loss, aux_last), g = grad_fn(params, *mbargs)
                    gnorm = float(optax.global_norm(g))
                    if not np.isfinite(gnorm):      # never apply a NaN update
                        stop = True
                        break
                    # Trust region, enforced by REVERTING (B1 consequence).
                    # The MPC mean is far more sensitive to the cost map than an
                    # MLP's output is to its weights: measured, one clipped Adam
                    # step moves mu by 0.32 for a 0.5 % change in S, which at
                    # sigma = 0.05 is already KL 4.8.  Checking KL only after
                    # the step -- the usual PPO early stop -- lets that step
                    # land, and the next iteration goes NaN.  So the step is
                    # applied, measured, and undone if it left the region.
                    prev = jax.tree_util.tree_map(lambda z: z, params)
                    prev_opt = opt_state
                    upd, opt_state = opt.update(g, opt_state, params)
                    params = optax.apply_updates(params, upd)
                    kl_new = float(loss_fn(params, *mbargs)[1][3])
                    if not np.isfinite(kl_new) or kl_new > 4 * c["kl_target"]:
                        params, opt_state = prev, prev_opt
                        aux_last = (aux_last[0], aux_last[1], aux_last[2],
                                    jnp.asarray(kl_new), aux_last[4])
                        lr_cur = max(lr_cur * c["lr_decay"], c["lr_min"])
                        opt_state.hyperparams["lr"] = jnp.asarray(lr_cur)
                        stop = True
                        break
                    kl_seen = max(kl_seen, kl_new)
        # B4: the observation statistics are refreshed at the END of the
        # iteration, never between the rollout and the update.  Updating them in
        # between makes mpc_mean in the loss normalise differently from the
        # rollout that produced lp_old, so the ratio is comparing two different
        # policies: measured, KL jumps to 2.1e4 on the first minibatch and the
        # parameters go NaN one iteration later.
        if "obs_norm" in params["actor"]:
            params["actor"] = dict(
                params["actor"],
                obs_norm=obs_norm_update(params["actor"]["obs_norm"], F["o"]))
        if kl_seen and kl_seen < 0.5 * c["kl_target"]:     # slack -> step up
            lr_cur = min(lr_cur * c["lr_grow"], c["lr"])
            opt_state.hyperparams["lr"] = jnp.asarray(lr_cur)
        rows.append(dict(iter=it, reward=float(ST["rw"].mean()), ep_len=float(T_rollout),
                         value_loss=float(aux_last[1]), policy_loss=float(aux_last[0]),
                         entropy=float(aux_last[2]), kl=float(aux_last[3]),
                         clipfrac=float(aux_last[4]), grad_norm=gnorm,
                         sigma=float(jnp.exp(params["log_sigma"]).mean()),
                         lr=lr_cur, sat=float(np.mean(sat_acc)),
                         crash_rate=float(np.mean(crash_acc)), wall_s=time.time()))
        if verbose and (it % max(iters // 10, 1) == 0 or it == iters - 1):
            r_ = rows[-1]
            print(f"  it {it:4d}  R {r_['reward']:+8.3f}  vloss {r_['value_loss']:8.3f} "
                  f"kl {r_['kl']:.4f}  lr {r_['lr']:.2e}  sat {r_['sat']:.3f}  "
                  f"crash {r_['crash_rate']:.3f}")
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


def _flat(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.concatenate([jnp.ravel(l) for l in leaves])


def _unflat(tree, vec):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    out, i = [], 0
    for l in leaves:
        n = l.size
        out.append(vec[i:i + n].reshape(l.shape))
        i += n
    return jax.tree_util.tree_unflatten(treedef, out)


def _cg(Avp, b, iters=10, tol=1e-10):
    """Conjugate gradient for the natural-gradient step (Schulman et al. 2015)."""
    x = jnp.zeros_like(b)
    r = b
    p = b
    rr = r @ r

    def body(carry, _):
        x, r, p, rr = carry
        Ap = Avp(p)
        alpha = rr / jnp.maximum(p @ Ap, 1e-20)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = r @ r
        p = r + (rr_new / jnp.maximum(rr, 1e-20)) * p
        return (x, r, p, rr_new), None

    (x, _, _, _), _ = jax.lax.scan(body, (x, r, p, rr), None, length=iters)
    return x


def trpo_step(actor, log_sigma, act_mu, ob, zz, lp_old, adv, max_kl=0.01,
              damping=0.1, backtracks=10):
    """One TRPO update: natural-gradient direction, then a backtracking line
    search that enforces the hard KL constraint and surrogate improvement.

    The Gaussian policy has a fixed diagonal sigma here, so the KL between old
    and new reduces to ||mu - mu_old||^2 / (2 sigma^2) and its Hessian is
    diag(1/sigma^2) -- the Fisher-vector product is then exact rather than
    estimated, which removes the usual source of TRPO flakiness.
    """
    inv_var = jnp.exp(-2.0 * log_sigma)

    def surrogate(p):
        mu = act_mu(p, ob)
        lp = _logp(zz, mu, log_sigma)
        return jnp.mean(jnp.exp(lp - lp_old) * adv)

    def kl(p):
        mu = act_mu(p, ob)
        mu_old = jax.lax.stop_gradient(act_mu(actor, ob))
        return 0.5 * jnp.mean(jnp.sum((mu - mu_old) ** 2 * inv_var, -1))

    g = _flat(jax.grad(surrogate)(actor))
    if float(jnp.linalg.norm(g)) < 1e-10:
        return actor, 0.0, 0.0

    def Avp(v):
        hv = jax.jvp(lambda p: _flat(jax.grad(kl)(p)), (actor,),
                     (_unflat(actor, v),))[1]
        return hv + damping * v

    step_dir = _cg(Avp, g)
    shs = 0.5 * step_dir @ Avp(step_dir)
    step = step_dir * jnp.sqrt(max_kl / jnp.maximum(shs, 1e-20))
    old_s = float(surrogate(actor))
    for i in range(backtracks):
        frac = 0.5 ** i
        cand = _unflat(actor, _flat(actor) + frac * step)
        new_s, new_kl = float(surrogate(cand)), float(kl(cand))
        if new_kl <= 1.5 * max_kl and new_s > old_s:
            return cand, new_kl, new_s - old_s
    return actor, 0.0, 0.0


def mlp_policy_init(key, obs_dim, hid, n_layer=2, normalise=True):
    """The model-free arm.  It gets the SAME observation normalisation as the
    cost map, or §8.3's exploration comparison would be measuring the
    normalisation rather than the architecture (B4)."""
    p = {"net": mlp_init(key, [obs_dim] + [hid] * n_layer + [NU], scale_last=0.01)}
    if normalise:
        p["obs_norm"] = obs_norm_init(obs_dim)
    return p


def make_mlp_ctrl(actor, name="MLP"):
    """The **model-free control arm** of §8.3 sweep 2.

    Identical reward, identical observation, identical optimiser -- only the
    differentiable optimiser is removed.  It exists to test the *mechanism*:
    additive noise on a box-constrained collective does not merely perturb the
    action, it drives the input onto the box and corrupts the linearisation
    A_k, B_k inside the solve.  A policy with no solve inside it cannot suffer
    that, so the two must be compared on the same noise.
    """
    @jax.jit
    def _act(o, uref):
        return jnp.clip(uref + jnp.tanh(mlp_apply(actor["net"],
                                                  normalise_obs(actor, o))) * 0.5,
                        jnp.asarray(U_LO), jnp.asarray(U_HI))

    def f(o, e, xr, uref=None, d=None):
        if uref is None:
            uref = jnp.zeros((o.shape[0], NU)).at[:, 0].set(U_HOVER)
        return _act(o, uref), {}
    f.name = name
    return f


def train_mlp(env, cfg, seed=0, iters=100, T_rollout=64, verbose=True,
              log_path=None):
    """PPO on a direct-action MLP: the model-free arm, same reward as (5.4)."""
    import optax
    import pandas as pd
    c = dict(PPO_DEFAULTS); c.update(cfg)
    key = jax.random.PRNGKey(seed)
    key, ka, kc = jax.random.split(key, 3)
    actor = mlp_policy_init(ka, env.obs_dim, c["hid"])
    critic = critic_init(kc, env.obs_dim, c["hid"])
    log_sigma = jnp.full((NU,), np.log(c["sigma"]))
    params = {"actor": actor, "critic": critic}
    opt = optax.chain(optax.clip_by_global_norm(c["max_grad_norm"]),
                      optax.adam(c["lr"]))
    opt_state = opt.init(params)

    @jax.jit
    def act(p, o):
        return jnp.tanh(mlp_apply(p["net"], normalise_obs(p, o))) * 0.5

    rows = []
    o, e, xr = env.obs()
    for it in range(iters):
        O, A, LP, RW, VL, DN, sat_acc, cr_acc = [], [], [], [], [], [], [], []
        for t in range(T_rollout):
            key, k = jax.random.split(key)
            mu = act(params["actor"], o)
            a = mu + jnp.exp(log_sigma) * jax.random.normal(k, mu.shape)
            _, uref, _ = env.ref_now()
            u = jnp.clip(uref + a, jnp.asarray(U_LO), jnp.asarray(U_HI))
            v = mlp_apply(params["critic"],
                          normalise_obs(params["actor"], o))[:, 0]
            r, done, info = env.step(u)
            O.append(o); A.append(a); LP.append(_logp(a, mu, log_sigma))
            RW.append(r); VL.append(v); DN.append(done.astype(jnp.float64))
            sat_acc.append(float(info["sat"])); cr_acc.append(float(jnp.mean(info["crash"])))
            o, e, xr = env.obs()
        v_last = mlp_apply(params["critic"], normalise_obs(params["actor"], o))[:, 0]
        O, A, LP = jnp.stack(O), jnp.stack(A), jnp.stack(LP)
        RW, VL, DN = jnp.stack(RW), jnp.stack(VL), jnp.stack(DN)
        ADV = gae(RW, VL, v_last, DN, c["gamma"], c["lam"])
        RET = ADV + VL
        flat = lambda z: z.reshape((-1,) + z.shape[2:])
        Of, Af, LPf, ADVf, RETf = map(flat, (O, A, LP, ADV, RET))
        ADVf = (ADVf - ADVf.mean()) / (ADVf.std() + 1e-8)

        def loss_fn(p, ob, aa, lp_old, adv, ret):
            mu = act(p["actor"], ob)
            lp = _logp(aa, mu, log_sigma)
            ratio = jnp.exp(lp - lp_old)
            l_pi = -jnp.mean(jnp.minimum(
                ratio * adv, jnp.clip(ratio, 1 - c["clip"], 1 + c["clip"]) * adv))
            l_v = jnp.mean((mlp_apply(p["critic"],
                                      normalise_obs(p["actor"], ob))[:, 0] - ret) ** 2)
            return l_pi + c["vf_coef"] * l_v, (l_pi, l_v, jnp.mean(lp_old - lp))

        gfn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        n = Of.shape[0]
        mb = max(n // c["minib"], 1)
        aux = None
        for _ in range(c["epochs"]):
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, n)
            for i in range(c["minib"]):
                idx = perm[i * mb:(i + 1) * mb]
                if idx.size == 0:
                    continue
                (_, aux), g = gfn(params, Of[idx], Af[idx], LPf[idx],
                                  ADVf[idx], RETf[idx])
                upd, opt_state = opt.update(g, opt_state, params)
                params = optax.apply_updates(params, upd)
        if "obs_norm" in params["actor"]:            # end of iteration, as in PPO
            params["actor"] = dict(params["actor"],
                                   obs_norm=obs_norm_update(
                                       params["actor"]["obs_norm"], Of))
        rows.append(dict(iter=it, reward=float(RW.mean()),
                         value_loss=float(aux[1]), policy_loss=float(aux[0]),
                         kl=float(aux[2]), sat=float(np.mean(sat_acc)),
                         crash_rate=float(np.mean(cr_acc))))
        if verbose and (it % max(iters // 5, 1) == 0 or it == iters - 1):
            print(f"  MLP it {it:4d}  R {rows[-1]['reward']:+8.3f}  "
                  f"sat {rows[-1]['sat']:.3f}")
    df = pd.DataFrame(rows)
    if log_path:
        df.to_csv(apath(*log_path), index=False)
    return params["actor"], params["critic"], df
