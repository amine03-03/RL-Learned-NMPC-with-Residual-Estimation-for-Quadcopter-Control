"""The ROS 2 node wiring, checked against the study without a ROS graph.

Every node is a thin wrapper around a pure core.  These tests pin each core to
the study function it must reproduce, so a deployment that drifts from what the
policy and the RDP were trained on fails here rather than in flight:

* the reference and u_ref the controller builds  == x500_core_jax.ref_state
* the (5.3) observation it feeds AC-MPC           == Env.obs()
* the RDP frame built from /acmpc/status          == x500_core_jax._frame26
* PX4 odometry (NED/FRD) -> ENU/FLU               round trip
* S1 / S5 as wrenches, and the Gazebo increment scheme
"""
import os
import sys

import numpy as np
import pytest

import x500_core_jax as X

WS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "rdp_acmpc_ws", "src")
for pkg in ("acmpc_controller", "rdp_estimator", "disturbance_manager",
            "reference_generator"):
    p = os.path.join(WS, pkg)
    if p not in sys.path:
        sys.path.insert(0, p)

from acmpc_controller import frames as F                     # noqa: E402
from acmpc_controller import status_msg                      # noqa: E402
from acmpc_controller.reference import ObsBuilder, ref_sequence  # noqa: E402
from disturbance_manager import scenarios as S               # noqa: E402
from rdp_estimator.ring_buffer import build_frame, quat_to_R  # noqa: E402


def _env(seed=3, task="track"):
    return X.Env(1, seed, 100000, X.nominal_spec(speed=(1.2, 1.2)), ("circle",),
                 task=task)


def _pva(ep):
    def f(t):
        t = np.atleast_1d(np.asarray(t, float))
        out = [X._ref_pva(ep, np.array([tk])) for tk in t]
        return tuple(np.concatenate([np.asarray(o[i]) for o in out], 0) for i in range(3))
    return f


# --------------------------------------------------------------------------- #
# reference and observation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hold", [False, True])
def test_ref_sequence_equals_study_ref_state(hold):
    env = _env()
    ts = 0.37 + np.arange(11) * X.P.dt_c
    xr, ur = ref_sequence(_pva(env.ep), ts, hold)
    for k, tk in enumerate(ts):
        xs, us, _ = X.ref_state(env.ep, np.array([tk]), hold=hold)
        np.testing.assert_allclose(xr[k], np.asarray(xs)[0], atol=1e-10)
        np.testing.assert_allclose(ur[k], np.asarray(us)[0], atol=1e-10)


def test_u_ref_hover_has_the_idle_floor():
    """A2: hover collective 0.7287, not the bare sqrt(mg/T_max) = 0.7694."""
    hold = lambda t: (np.tile([0, 0, 1.5], (np.size(t), 1)),
                      np.zeros((np.size(t), 3)), np.zeros((np.size(t), 3)))
    _, ur = ref_sequence(hold, np.zeros(1), hold=True)
    assert abs(ur[0, 0] - X.U_HOVER) < 1e-9
    assert abs(ur[0, 0] - np.sqrt(X.M_TOT * X.P.g / X.T_MAX)) > 0.03


def test_obs_builder_reproduces_env_obs():
    """Drive the study Env with random commands; the node's ObsBuilder, fed what
    the node can see (error, body rate, its own past command), must produce the
    same 40-D observation at every step."""
    env = _env(seed=5)
    rng = np.random.default_rng(0)
    ob = ObsBuilder(_pva(env.ep), hold=False)
    u_prev = np.array([X.U_HOVER, 0.0, 0.0, 0.0])
    uref_prev = np.array([X.U_HOVER, 0.0, 0.0, 0.0])
    for k in range(25):
        o, e, xr = env.obs()
        t = float(np.asarray(env.t)[0])
        _, uref, _ = X.ref_state(env.ep, np.array([t]))
        if k > 0:
            ob.update(np.asarray(e)[0], np.asarray(env.state)[0, X.SW],
                      u_prev - uref_prev, u_prev)
        mine = ob.obs(np.asarray(e)[0], np.asarray(env.state)[0, X.SP], t)
        np.testing.assert_allclose(mine, np.asarray(o)[0], atol=1e-9,
                                   err_msg=f"step {k}")
        u = np.clip(np.asarray(uref)[0] + rng.normal(0, 0.05, 4), X.U_LO, X.U_HI)
        env.step(u[None])
        u_prev, uref_prev = u, np.asarray(uref)[0]


# --------------------------------------------------------------------------- #
# the RDP frame and the status message
# --------------------------------------------------------------------------- #
def test_status_roundtrip():
    rng = np.random.default_rng(1)
    kw = {n: rng.normal(size=w) if w > 1 else float(rng.normal())
          for n, w in status_msg.FIELDS}
    back = status_msg.unpack(status_msg.pack(**kw))
    assert status_msg.SIZE == 38
    for n, _ in status_msg.FIELDS:
        np.testing.assert_allclose(back[n], kw[n])


def test_frame_from_status_equals_study_frame26():
    """What the estimator node builds from /acmpc/status + actuator_motors must
    be the study's _frame26 (6.2): same order, row-major vec(R), same units."""
    env = _env(seed=7)
    rng = np.random.default_rng(2)
    for _ in range(6):
        env.step(np.clip([[X.U_HOVER, 0, 0, 0]] + rng.normal(0, 0.05, (1, 4)),
                         X.U_LO, X.U_HI))
    s = np.asarray(env.state)[0]
    t = float(np.asarray(env.t)[0])
    u_prev = np.asarray(env.prev["u"])[0]
    xr, uref, _ = X.ref_state(env.ep, np.array([t]))
    want = np.asarray(X._frame26(env.state, env.ep, env.t, env.prev["u"], uref, env.par))[0]
    _, pwm, _ = X.inner_loop(env.state, env.prev["u"], env.par)
    st = status_msg.unpack(status_msg.pack(
        t=t, p=s[X.SP], v=s[X.SV], q=s[X.SQ], om=s[X.SW], p_ref=np.asarray(xr)[0, 0:3],
        v_ref=np.asarray(xr)[0, 3:6], q_ref=np.asarray(xr)[0, 6:10], u_prev=u_prev,
        u_ref=np.asarray(uref)[0], u=u_prev, solve_ms=0.0, loop_ms=0.0))
    got = build_frame(st["p"], st["p_ref"], quat_to_R(st["q"]), st["v"], st["om"],
                      st["u_prev"], st["u_ref"], np.asarray(pwm)[0])
    np.testing.assert_allclose(got, want, atol=1e-12)


# --------------------------------------------------------------------------- #
# PX4 odometry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("frame", [F.VELOCITY_FRAME_NED, F.VELOCITY_FRAME_BODY_FRD])
def test_odometry_roundtrip(frame):
    rng = np.random.default_rng(3)
    for _ in range(50):
        p, v, om = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
        q = F.qnormalise(rng.normal(size=4))
        q = q if q[0] >= 0 else -q
        vel = (F.enu_to_ned_vec(v) if frame == F.VELOCITY_FRAME_NED
               else F.flu_to_frd_vec(F.qrot(F.qconj(q), v)))
        p2, v2, q2, om2 = F.odometry_to_enu_flu(
            F.enu_to_ned_vec(p), F.enu_flu_to_px4_quat(q), vel,
            F.flu_to_frd_vec(om), frame)
        q2 = q2 if q2[0] >= 0 else -q2
        for a, b in ((p, p2), (v, v2), (q, q2), (om, om2)):
            np.testing.assert_allclose(a, b, atol=1e-12)


def test_odometry_refuses_unknown_velocity_frame():
    with pytest.raises(ValueError):
        F.odometry_to_enu_flu(np.zeros(3), [1, 0, 0, 0], np.zeros(3), np.zeros(3), 2)


# --------------------------------------------------------------------------- #
# scenarios as wrenches, and the Gazebo increment scheme
# --------------------------------------------------------------------------- #
def test_rotor_constants_match_study():
    assert S.K_T == X.P.K_T and S.K_M == X.P.k_m
    assert S.OM_MIN == X.P.Om_min and S.OM_MAX == X.P.Om_max
    np.testing.assert_array_equal(S.R_ROTOR, X.R_ROTOR)
    np.testing.assert_array_equal(S.SIGMA, X.SIGMA)


def test_S1_payload_is_the_C6_standing_moment_at_hover():
    s = S.Scenario("S1", seed=4)
    F_w, tau = s.wrench(0.5 * (s.t_on + s.t_off))
    m_p = (s.params["mass_scale"] - 1.0) * S.M_NOM
    d = np.asarray(s.params["cg_offset"])
    np.testing.assert_allclose(F_w, [0, 0, -m_p * S.G])
    np.testing.assert_allclose(tau, -m_p * S.G * np.array([d[1], -d[0], 0.0]), atol=1e-15)
    assert np.all(s.wrench(s.t_on - 0.1)[0] == 0) and np.all(s.wrench(s.t_off)[0] == 0)


def test_S5_motor_loss_matches_allocation():
    """At hover, losing (1-eta) of rotor i's thrust removes exactly that rotor's
    column of the allocation matrix, scaled by the lost thrust."""
    s = S.Scenario("S5", seed=1, params=dict(motor_index=2, motor_effectiveness=0.9))
    pwm = np.full(4, X.U_HOVER)
    F_w, tau = s.wrench(6.0, R=np.eye(3), pwm=pwm)
    om = S.OM_MIN + X.U_HOVER * (S.OM_MAX - S.OM_MIN)
    dT = 0.1 * S.K_T * om ** 2
    col = X._alloc(X.R_ROTOR, X.P.k_m, X.SIGMA)[:, 2]          # [1, y, -x, -sigma k_m]
    np.testing.assert_allclose(F_w, [0, 0, -dT], atol=1e-12)
    np.testing.assert_allclose(tau, -dT * col[1:], atol=1e-12)
    assert np.all(s.wrench(6.0, R=np.eye(3), pwm=None)[0] == 0)   # no PWM, no S5


def test_persistent_wrench_sum_tracks_the_target():
    rng = np.random.default_rng(5)
    pw = S.PersistentWrench(resync_every=7)
    applied = np.zeros(6)
    for k in range(60):
        want = rng.normal(size=6) if k % 3 else applied.copy()
        act, d = pw.step(want[:3], want[3:])
        if act == "clear":
            applied = np.zeros(6)
            continue
        if act == "add":
            applied = applied + d
        np.testing.assert_allclose(applied, want, atol=1e-12)


def test_to_world_rotates_the_torque_only():
    R = F.qrotmat(F.qnormalise(np.array([0.9, 0.1, -0.3, 0.2])))
    Fw, tw = S.to_world([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], R)
    np.testing.assert_allclose(Fw, [1, 2, 3])
    np.testing.assert_allclose(tw, R @ [0.1, 0.2, 0.3])


# --------------------------------------------------------------------------- #
# the controller core end to end
# --------------------------------------------------------------------------- #
def test_controller_on_reference_returns_u_ref():
    """On the reference with no disturbance the NMPC must return the feed
    forward exactly -- i.e. the reference is consistent with the model."""
    from acmpc_controller.controller import ACMPCController
    from reference_generator import lissajous as L
    pva = lambda t: L.reference(t, "moderate", 1.2, 0.9, 1.5)
    c = ACMPCController(mode="nmpc1", horizon=5, n_iter=5, dmod_mode="closed_loop")
    c.set_reference(pva, False)
    xr, ur = ref_sequence(pva, np.zeros(1))
    u, info = c.tick(xr[0], np.zeros(3), 0.0)
    # iLQR stops after n_iter iterations, so "exactly" means to ~1e-3
    np.testing.assert_allclose(u, ur[0], atol=2e-3)
    np.testing.assert_allclose(info["u_ref"], ur[0], atol=1e-12)


# --------------------------------------------------------------------------- #
# the hand-over entry trajectory
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("level", ["moderate", "hold"])
def test_entry_reference_joins_the_path_smoothly(level):
    from acmpc_controller.reference import entry_reference
    from reference_generator import lissajous as L
    base = lambda t: L.reference(t, level, 1.2, 0.9, 1.5)
    p0, v0 = np.array([0.1, -0.2, 2.5]), np.array([0.05, 0.0, -0.1])
    pva, T = entry_reference(base, p0, v0, T=4.0, a_max=3.0)
    p, v, a = pva(np.array([0.0]))
    np.testing.assert_allclose(p[0], p0, atol=1e-12)
    np.testing.assert_allclose(v[0], v0, atol=1e-12)
    np.testing.assert_allclose(a[0], 0.0, atol=1e-12)
    for x_e, x_b in zip(pva(np.array([T - 1e-9])), base(np.zeros(1))):
        np.testing.assert_allclose(x_e[0], x_b[0], atol=1e-6)     # C2 at the join
    ts = np.array([T, T + 0.7, T + 3.1])
    for x_e, x_b in zip(pva(ts), base(ts - T)):
        np.testing.assert_allclose(x_e, x_b, atol=1e-12)          # then the path
    acc = pva(np.linspace(0, T, 400))[2]
    assert np.linalg.norm(acc, axis=1).max() <= 3.0 + 1e-9         # within budget


def test_entry_reference_starts_at_the_vehicle_so_the_first_command_is_benign():
    """The SITL failure: handed straight to the path, NMPC commanded zero
    collective.  From the entry reference the first command is near hover."""
    from acmpc_controller.controller import ACMPCController
    from acmpc_controller.reference import entry_reference
    from reference_generator import lissajous as L
    base = lambda t: L.reference(t, "moderate", 1.2, 0.9, 1.5)
    x10 = np.array([0.0, 0.0, 2.5, 0, 0, 0, 1, 0, 0, 0])
    c = ACMPCController(mode="nmpc1", horizon=10, n_iter=10, dmod_mode="closed_loop")
    c.set_reference(base, False)
    u_step, _ = c.tick(x10, np.zeros(3), 0.0)
    pva, _ = entry_reference(base, x10[0:3], np.zeros(3))
    c.set_reference(pva, False)
    c.last_u = np.array([X.U_HOVER, 0.0, 0.0, 0.0])
    u_entry, _ = c.tick(x10, np.zeros(3), 0.0)
    assert u_step[0] < 0.1                         # the failure mode, reproduced
    assert abs(u_entry[0] - X.U_HOVER) < 0.05      # hover-like collective
    assert np.all(np.abs(u_entry[1:]) < 0.05)      # no saturated rates


def test_delay_compensation_predicts_with_the_study_model():
    """delay_steps = k propagates the measured state through the k commands in
    flight with x500_core_jax.step_c, and shifts the reference by k periods."""
    import jax.numpy as jnp
    from acmpc_controller.controller import ACMPCController
    c = ACMPCController(mode="nmpc1", horizon=5, n_iter=5, dmod_mode="closed_loop",
                        use_d=False, delay_steps=2)
    x = np.array([0.1, -0.2, 1.4, 0.3, 0.0, -0.1, 0.995, 0.05, -0.03, 0.08])
    x = np.concatenate([x[:6], x[6:] / np.linalg.norm(x[6:])])
    u1, u2 = np.array([0.75, 0.02, -0.01, 0.0]), np.array([0.70, -0.03, 0.02, 0.01])
    c._inflight = [u1, u2]
    want = X.step_c(X.step_c(jnp.asarray(x)[None], jnp.asarray(u1)[None]),
                    jnp.asarray(u2)[None])
    np.testing.assert_allclose(c.predict(x), np.asarray(want)[0], atol=1e-12)
    c0 = ACMPCController(mode="nmpc1", horizon=5, n_iter=5, dmod_mode="closed_loop")
    np.testing.assert_array_equal(c0.predict(x), x)                # off by default
