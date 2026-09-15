"""T-13 .. T-20 -- GAE/MPVE, moderation, clips, checkpoints, parity, frames,
tether energy and the scale guard."""
import os
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import adaptive_core_jax as A
import study_moderate as M
import viz as V
import x500_core_jax as X


def test_T13_gae_closed_form():
    """T-13: GAE against the closed form on a 5-step episode, exactly."""
    g, lam = 0.99, 0.95
    rew = jnp.asarray([[1.0], [2.0], [-1.0], [0.5], [3.0]])
    val = jnp.asarray([[0.4], [0.3], [0.2], [0.1], [0.0]])
    v_last = jnp.asarray([0.7])
    done = jnp.zeros((5, 1))
    got = np.asarray(X.gae(rew, val, v_last, done, g, lam))[:, 0]
    r = np.asarray(rew)[:, 0]; v = np.asarray(val)[:, 0]
    vn = np.r_[v[1:], float(v_last[0])]
    delta = r + g * vn - v
    want = np.zeros(5)
    acc = 0.0
    for k in range(4, -1, -1):
        acc = delta[k] + g * lam * acc
        want[k] = acc
    assert np.abs(got - want).max() < 1e-12


def test_T13_gae_respects_termination():
    """A done flag must cut the bootstrap; otherwise value leaks across episodes."""
    g, lam = 0.99, 0.95
    rew = jnp.asarray([[1.0], [1.0], [1.0]])
    val = jnp.asarray([[0.5], [0.5], [0.5]])
    done = jnp.asarray([[0.0], [1.0], [0.0]])
    got = np.asarray(X.gae(rew, val, jnp.asarray([0.5]), done, g, lam))[:, 0]
    assert abs(got[1] - (1.0 - 0.5)) < 1e-12, "done did not cut the bootstrap"


def test_T13_mpve_three_step():
    """T-13: MPVE (5.9) against a hand-computed 3-step case."""
    g = 0.9
    B, N = 1, 3
    e_seq = jnp.zeros((B, N + 1, X.NE)).at[:, :, 0].set(
        jnp.asarray([0.3, 0.2, 0.1, 0.05]))
    du = jnp.zeros((B, N, X.NU))
    om = jnp.zeros((B, 3))
    d_u = jnp.zeros((B, X.NU))
    critic = [(jnp.zeros((X.NE, 1)), jnp.asarray([2.0]))]        # constant V = 2
    got = float(X.mpve_targets(e_seq, du, critic, g, om, d_u)[0])
    want = 0.0
    for k in range(N):
        want += g ** k * float(X.reward_quad(
            e_seq[:, k + 1], du[:, k], om, d_u, jnp.zeros(B))[0])
    want += g ** N * 2.0
    assert abs(got - want) < 1e-12


def test_T14_moderate_roundtrip():
    """T-14: containment (as N-6 states it), narrowing, speed untouched."""
    for raw, wind in ((X.disturbed_spec(), (0.0, 2.0)), (X.ood_spec(), (2.5, 4.0))):
        mod = M.moderate(raw, wind=wind)
        assert M.check_moderate(raw, mod, wind[1])
        assert mod["speed"] == raw["speed"]
        for k in ("m", "D", "tau", "T", "Kw", "J"):
            assert abs((mod[k][1] - mod[k][0]) / (raw[k][1] - raw[k][0]) - 0.15) < 1e-9
    # the §7.2 table, reproduced
    m2 = M.moderate(X.disturbed_spec())
    assert abs(m2["m"][0] - 0.970) < 1e-9 and abs(m2["m"][1] - 1.030) < 1e-9
    assert abs(m2["D"][0] - 0.925) < 1e-9 and abs(m2["D"][1] - 1.150) < 1e-9
    assert abs(m2["tau"][1] - 1.300) < 1e-9


@pytest.mark.parametrize("key,bad", [
    ("speed", (0.1, 0.2)), ("D", (1.02, 1.15)), ("m", (0.5, 1.5))])
def test_T14_check_moderate_raises_naming_the_key(key, bad):
    raw = X.disturbed_spec()
    mod = {**M.moderate(raw), key: bad}
    with pytest.raises(AssertionError) as ex:
        M.check_moderate(raw, mod, 2.0)
    assert key in str(ex.value)


def test_T14b_disjointness_claim_C4():
    """C-4: the moderated suites are disjoint on drag, lag and wind ONLY.

    Mass, thrust, K_w and inertia all overlap, because the raw S3 band is a
    superset of the raw S2 band in each and contraction preserves inclusion.
    """
    d = M.disjointness(M.moderate(X.disturbed_spec()),
                       M.moderate(X.ood_spec(), wind=(2.5, 4.0)))
    disj = set(d.loc[d.disjoint, "channel"])
    assert disj == {"D", "tau", "wind"}, f"disjoint set is {sorted(disj)}"
    over = set(d.loc[~d.disjoint, "channel"])
    assert over == {"m", "T", "Kw", "J"}, f"overlapping set is {sorted(over)}"
    assert d.loc[~d.disjoint, "S2_subset_of_S3"].all()


def test_T15_clip_coverage():
    """T-15: theta span >= 2pi, RFULL closed, axis box contains RFULL,
    respawns == 0."""
    env = X.Env(2, 4, 5000, X.nominal_spec(speed=(1.0, 1.0)), ("circle",),
                fixed=dict(R=1.0))
    T, per = V.clip_length(env, 1.0)
    assert per is not None and T * X.P.dt_c >= per - 1e-9
    d = V.fly(env, X.make_lqr_ctrl(), T)
    assert d["respawns"] == 0
    # duration coverage is the clip property; the flown angular span is only
    # expected to follow for a controller that actually held the path, which
    # the LQR does on a nominal circle
    assert T * X.P.dt_c >= per - 1e-9
    assert d["rmse"] < 0.5, "this check assumes the pilot held the path"
    assert d["theta_span"] >= 2 * np.pi - 1e-3
    assert np.linalg.norm(d["RFULL"][0] - d["RFULL"][-1]) < 1e-6
    lo, hi = d["RFULL"].min(0), d["RFULL"].max(0)
    ctr, half = (lo + hi) / 2, max((hi - lo).max(), 1.0) / 2 * 1.25
    assert bool(((d["RFULL"] >= ctr - half - 1e-9)
                 & (d["RFULL"] <= ctr + half + 1e-9)).all())


def test_T15_period_is_not_a_fixed_step_count():
    """A fixed step count cannot span a period, because omega depends on the
    sampled radius through (4.6)."""
    Ts = []
    for R in (0.5, 1.0, 2.0):
        env = X.Env(1, 0, 5000, X.nominal_spec(speed=(1.5, 1.5)), ("square",),
                    fixed=dict(R=R))
        Ts.append(V.clip_length(env, 1.0)[0])
    assert len(set(Ts)) == len(Ts), f"clip length is not radius dependent: {Ts}"


def test_T16_checkpoint_round_trip(tmp_path):
    """T-16: round trip for every artefact type on disk, exactly."""
    key = jax.random.PRNGKey(0)
    objs = {
        "costmap": X.costmap_init(key, X.OBS_DIM, 32, "diag", 1),
        "critic": X.critic_init(key, X.OBS_DIM, 32),
        "rdp": A.rdp_init(key, "GRU", H=16, hid=(8, 4)),
        "dict": {"a": np.arange(5), "b": {"c": 1.5}},
    }
    os.environ["X500_ARTIFACTS"] = str(tmp_path)
    import importlib
    importlib.reload(X)
    for name, o in objs.items():
        X.save_ckpt(o, "test", f"{name}.pkl")
        back = X.load_ckpt("test", f"{name}.pkl")
        fa = jax.tree_util.tree_leaves(o)
        fb = jax.tree_util.tree_leaves(back)
        assert len(fa) == len(fb), name
        for a, b in zip(fa, fb):
            assert np.array_equal(np.asarray(a), np.asarray(b)), name
    X.save_json({"x": 1, "y": [1.0, 2.0]}, "test", "j.json")
    assert X.load_json("test", "j.json") == {"x": 1, "y": [1.0, 2.0]}


@pytest.mark.parametrize("kind", A.ENCODERS)
def test_T17_numpy_jax_parity(kind, tmp_path):
    """T-17: rdp_infer (NumPy) vs the JAX predictor, 64 random windows, 1e-5."""
    import export_estimator as E
    key = jax.random.PRNGKey(1)
    p = A.rdp_init(key, kind, H=32)
    mu, sd = np.zeros(A.FRAME_DIM), np.ones(A.FRAME_DIM)
    out_sd = np.array([2.0, 2.0, 2.0, 0.2, 0.2, 0.2])
    entry = A.ckpt_entry("x", p, kind, {}, mu, sd, out_sd, 32, [0.0] * 6)["x"]
    res = E.export(entry, str(tmp_path / f"rdp_{kind}.npz"), bench=False)
    assert res["parity"] < 1e-5


def test_T18_frames():
    """T-18: frame round trip, CTBR mapping, T_max agreement across modules."""
    from acmpc_controller import frames as F
    rng = np.random.default_rng(18)
    worst_q = worst_R = 0.0
    for _ in range(500):
        q = F.qnormalise(rng.normal(size=4))
        qe = F.px4_quat_to_enu_flu(q)
        back = F.enu_flu_to_px4_quat(qe)
        worst_q = max(worst_q, min(np.abs(back - q).max(), np.abs(back + q).max()))
        R = (F.qrotmat(F.Q_NED_ENU[None])[0] @ F.qrotmat(q[None])[0]
             @ F.qrotmat(F.Q_FRD_FLU[None])[0])
        worst_R = max(worst_R, np.abs(F.qrotmat(qe[None])[0] - R).max())
    assert worst_q < 1e-15
    # the matrix triple product accumulates a few ulp; the quaternion round trip
    # above is the 1e-15 statement, this is its matrix shadow
    assert worst_R < 5e-15
    # tilt is heading independent
    q0 = F.qnormalise(np.array([np.cos(0.2), np.sin(0.2), 0.0, 0.0]))
    tilts = [F.tilt_angle(F.qmul(np.array([np.cos(p / 2), 0, 0, np.sin(p / 2)]), q0))
             for p in np.linspace(0, 2 * np.pi, 64)]
    assert float(np.ptp(tilts)) < 1e-17
    # nine assertions on the CTBR exit mapping
    om = np.array([1.0, 2.0, 3.0])
    r = F.ctbr_to_px4_rates(om)
    assert r[0] == om[0] and r[1] == -om[1] and r[2] == -om[2]
    t = F.ctbr_to_px4_thrust(0.77)
    assert t[0] == 0.0 and t[1] == 0.0 and abs(t[2] + 0.77) < 1e-15
    assert np.allclose(F.ned_to_enu_vec(F.ned_to_enu_vec(om)), om)
    assert np.allclose(F.frd_to_flu_vec(F.frd_to_flu_vec(om)), om)
    # T_max derived from ONE module
    import check_consistency as C
    assert abs(C.STUDY_CONSTANTS["T_max"] - X.T_MAX) < 1e-12


def test_T19_tether_energy_drift():
    """T-19: slung-load tether energy drift over an episode < 2 %.

    Explicit Euler on an oscillator injects energy monotonically; the
    integration is velocity-first (semi-implicit) for exactly this reason.
    """
    s = A.SlungScenario(level=10.0)
    p = jnp.asarray([[np.sin(0.25), 0.0, -np.cos(0.25), 0.0, 0.0, 0.0]])
    E0 = float(s.energy(p)[0])
    for _ in range(1500):                                # 30 s, ~21 periods
        p = s.advance(p)
    assert abs(float(s.energy(p)[0]) - E0) / abs(E0) < 0.02


def test_T19b_tether_sign_C3():
    """C-3: a statically hanging load must pull the airframe DOWN."""
    s = A.SlungScenario(level=10.0)
    st = X.hover_state(1)
    F = np.asarray(s.wrench_body(jnp.asarray([[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]]),
                                 st, 0.0))[0, :3]
    assert F[2] < 0.0, f"hanging load pushes the airframe up: {F}"
    assert abs(F[2] + s.m_load * X.P.g) < 1e-9
    # the free-pendulum period
    assert abs(A.SLUNG_PERIOD - 2 * np.pi * np.sqrt(A.SLUNG_L / X.P.g)) < 1e-12


def test_T20_scale_guard_warns(capsys):
    """T-20: at `smoke`, the reporting layer detects the undertrained signature."""
    import pandas as pd
    import study_prelude as S
    df = pd.DataFrame(dict(ctrl=["LQR", "NMPC N=1", "AC-MPC", "LLTC N=1"],
                           rmse=[0.04, 0.05, 1.2, 0.9]))
    assert S.scale_guard(df) is True
    assert "UNDERTRAINED-POLICY SIGNATURE" in capsys.readouterr().out
    ok = pd.DataFrame(dict(ctrl=["LQR", "AC-MPC"], rmse=[0.04, 0.05]))
    assert S.scale_guard(ok) is False
