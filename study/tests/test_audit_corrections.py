"""Regression tests for the audit findings (docs/AUDIT.md).

Each asserts the corrected behaviour *and*, where it is cheap, the measurement
that showed the original was wrong.  These exist because every one of these
errors was silent: nothing raised, the numbers were simply wrong.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import adaptive_core_jax as A
import x500_core_jax as X


# --------------------------------------------------------------------------- #
# A. physics against the PX4 SDF and the gz_x500 airframe
# --------------------------------------------------------------------------- #
def test_A1_sdf_constants_exact():
    """Every value the SDF fixes must match it exactly, not approximately."""
    assert X.P.m_body == 2.0
    assert X.P.m_rotor == 0.016076923076923075
    assert X.P.K_T == 8.54858e-06
    assert X.P.k_m == 0.016                      # Gazebo momentConstant
    assert X.P.Om_max == 1000.0                  # maxRotVelocity / SIM_GZ_EC_MAX
    assert X.P.tau_up == 0.0125 and X.P.tau_dn == 0.0250
    assert X.P.J_body[0] == 0.02166666666666667
    rot = np.array([[.174, -.174, .06], [-.174, .174, .06],
                    [.174, .174, .06], [-.174, -.174, .06]])
    assert np.abs(X.R_ROTOR - rot).max() == 0.0
    assert np.array_equal(X.SIGMA, [1, 1, -1, -1])      # ccw, ccw, cw, cw


def test_A2_idle_floor():
    """SIM_GZ_EC_MIN = 150: thrust is NOT proportional to c^2."""
    assert X.P.Om_min == 150.0
    assert abs(X.U_HOVER - 0.728742) < 1e-6          # not 0.769431
    assert abs(X.DAZ_DC_HOVER - 21.666957) < 1e-5    # not 25.490538
    # a pure-square model would be 17.65 % optimistic about control authority
    assert abs(2 * X.T_MAX * 0.769431 / X.M_TOT / X.DAZ_DC_HOVER - 1.1765) < 1e-3
    # c = 0 is an idle floor, not zero thrust
    assert abs(float(X.thrust_of(jnp.zeros((1, 1)), X.T_MAX)[0, 0]) - 0.7694) < 1e-3
    assert abs(float(X.thrust_of(jnp.ones((1, 1)), X.T_MAX)[0, 0]) - X.T_MAX) < 1e-9
    assert abs(X.omega_of_cmd(X.cmd_of_omega(437.0)) - 437.0) < 1e-9


def test_A3_allocator_believes_a_different_k_m():
    """CA_ROTORn_KM = 0.05 vs the plugin's 0.016: a real 3.125x yaw error."""
    assert X.P.k_m_ctrl == 0.05
    assert abs(float(np.linalg.cond(X.M_CTRL)) - 20.0) < 1e-9
    assert abs(float(np.linalg.cond(X.M_NOM)) - 62.5) < 1e-9
    f = np.linalg.inv(X.M_CTRL) @ np.array([0.0, 0.0, 0.0, 0.1])
    realised = (X.M_NOM @ f)[3]
    assert abs(realised / 0.1 - 0.32) < 1e-9
    # and make_par must hand the plant the truth and the allocator the belief
    par = X.make_par(1)
    assert np.abs(np.asarray(par["Minv_ctrl"])[0] - X.MINV_CTRL).max() < 1e-12


def test_A4_pwm_is_normalised_on_the_px4_range():
    """/fmu/out/actuator_motors normalises on [Om_min, Om_max] (A4)."""
    par = X.make_par(1)
    s, u = X.hover_state(1, par=par), X.hover_u(1, par=par)
    _, pwm, _ = X.inner_loop(s, u, par)
    assert abs(float(pwm[0, 0]) - X.U_HOVER) < 1e-6      # 0.7287, not 0.7694
    assert abs(float(pwm[0, 0]) - float(X.cmd_of_omega(X.OM_HOVER))) < 1e-9


def test_A6_rotor_inertia_is_included():
    """The SDF gives each rotor an inertia tensor; point masses understate J."""
    pt = np.diag(X.P.J_body) + sum(
        X.P.m_rotor * (((r - X.CG_NOM) @ (r - X.CG_NOM)) * np.eye(3)
                       - np.outer(r - X.CG_NOM, r - X.CG_NOM)) for r in X.R_ROTOR)
    assert (np.diag(X.J_NOM) > np.diag(pt)).all()
    rel = (np.diag(X.J_NOM) - np.diag(pt)) / np.diag(pt)
    assert abs(rel[1] - 0.0043835) < 1e-5            # yy, the largest at 0.44 %


# --------------------------------------------------------------------------- #
# D7. the input box is the reachable set
# --------------------------------------------------------------------------- #
def test_D7_rate_box_is_reachable():
    """om_max is (10,10,4) but the autopilot clips at 3.84 rad/s.

    Constraining the box to the reachable set is what makes the control model
    and the plant agree; a clip inside fc() instead leaves a dead gradient
    beyond the knee and the collective absorbs the difference (measured: 47 %
    untrained saturation).
    """
    reach = X.P.rate_max / np.asarray(X.P.om_max)
    assert np.allclose(X.U_HI[1:], np.minimum(1.0, reach))
    assert np.allclose(X.U_LO[1:], -np.minimum(1.0, reach))
    assert X.U_LO[0] == 0.0 and X.U_HI[0] == 1.0
    # inside the box, the model's rate equals the plant's commanded rate
    for ub in (0.0, 0.2, float(X.U_HI[1])):
        u = jnp.asarray([[X.U_HOVER, ub, 0.0, 0.0]])
        assert abs(float(X.rate_cmd(u)[0, 0]) - ub * X.OM_MAX[0]) < 1e-9


def test_D7_no_saturation_at_trim():
    """With the box right, the hand-built controllers do not saturate."""
    env = X.Env(16, 1, 100000, X.nominal_spec(speed=(1.0, 1.5)), ("circle",))
    st = X.stats(X.rollout_eval(env, X.make_lqr_ctrl(), 120, warmup=30))
    assert st["sat"] < 0.02 and st["crash"] == 0.0


# --------------------------------------------------------------------------- #
# D3/D5. (4.13) validity, and the helix
# --------------------------------------------------------------------------- #
def test_D3_moment_conversion_is_exact_under_the_17_state_model():
    """(4.13) is gone: the moment block has an exact image now.

    The 10-state model had no torque input, so a wrench's moment could only be
    approximated by the rate loop's steady state with K_i neglected -- which
    measured 69-103x SMALLER than the (4.9) residual it was meant to cancel.
    With omega a state the conversion is alpha_res = J_nom^-1 tau, and that is
    EXACTLY the angular acceleration the plant shows at the instant the moment
    is applied.
    """
    assert X.DMOD_MODES == ("exact", "none")
    assert not hasattr(X, "moment_gain"), "moment_gain should be gone"
    assert not hasattr(X, "DMOD_SETTLE_S"), "DMOD_SETTLE_S should be gone"
    tau = 0.15
    w = jnp.asarray([[0.0, 0, 0, tau, 0, 0]])
    predicted = float(X.wrench_to_dmod(w)[0, 3])
    assert abs(predicted - tau / X.J_NOM[0, 0]) < 1e-12
    # against the plant, at t = 0, before the rate loop has responded
    par = X.par_set_wrench(X.make_par(1), w)
    sdot = X.fp(X.hover_state(1, par=par), X.hover_u(1, par=par), par)
    assert abs(float(sdot[0, X.SW][0]) - predicted) < 1e-9
    assert float(X.wrench_to_dmod(w, "none")[0, 3]) == 0.0
    with pytest.raises(ValueError):
        X.wrench_to_dmod(w, "first_order")


@pytest.mark.parametrize("R,spd", [(2.0, 6.5), (0.5, 6.5), (1.0, 1.5)])
def test_D5_helix_demand_matches_the_reference(R, spd):
    """path_demand must agree with _ref_pva even when (4.6)'s cap binds."""
    w = float(X.path_omega("helix", R, spd))
    ep = dict(kind=jnp.asarray([4]), c=jnp.zeros((1, 3)), R=jnp.asarray([R]),
              phi0=jnp.zeros(1), omega=jnp.asarray([w]), spd=jnp.asarray([spd]),
              delta=jnp.zeros((1, 3)))
    worst = 0.0
    for t in np.linspace(0.0, 4.0, 40):
        _, v, _ = X._ref_pva(ep, jnp.asarray([t]))
        worst = max(worst, float(jnp.linalg.norm(v)))
    assert abs(worst - X.path_demand("helix", R, w)[0]) < 1e-6


# --------------------------------------------------------------------------- #
# B. the AC-MPC learning stack against arXiv:2306.09852
# --------------------------------------------------------------------------- #
def test_B2_cost_map_learns_Q_and_p():
    """The paper's cost-map output is 2T(n+m): Q AND the linear term p."""
    assert X.REP_DIM["diag"] == 2 * X.NTAU
    for rep in X.REPS:
        assert X.REP_DIM[rep] == X.REP_Q_DIM[rep] + X.NTAU
    z = jnp.zeros((2, 1, X.REP_DIM["diag"])).at[:, :, X.NTAU:].set(3.0)
    S, c = X.costmap_from_z(z, "diag")
    assert float(jnp.abs(c).max()) > 0.1, "p is identically zero -- it is not learned"
    assert float(jnp.abs(c).max()) <= X.P_HI + 1e-9
    # p must not be able to dominate the quadratic term
    assert X.P_HI < np.sqrt(X.Q_LO * X.Q_HI)


def test_B4_observation_normalisation_exists_and_travels():
    th = X.costmap_init(jax.random.PRNGKey(0), X.OBS_DIM, 16, "diag", 1)
    assert "obs_norm" in th
    o = jax.random.normal(jax.random.PRNGKey(1), (64, X.OBS_DIM)) * 5.0 + 3.0
    th = dict(th, obs_norm=X.obs_norm_update(th["obs_norm"], o))
    n = X.normalise_obs(th, o)
    assert abs(float(n.mean())) < 0.2 and abs(float(n.std()) - 1.0) < 0.2


def test_B1_noise_is_on_the_action():
    """The policy must be Gaussian over the COMMAND, not over the cost map.

    The signature is that saturation rises with sigma: noise on a box-constrained
    collective drives the input onto the box, which is the mechanism §8.3's
    exploration study is about.  Parameter noise does not do that.
    """
    import inspect
    src = inspect.getsource(X.train_ppo)
    assert "a = mu + jnp.exp" in src, "the sample is not drawn around the MPC output"
    assert "_logp(a, mu" in src, "the log-prob is not taken over the action"
    assert "log_sigma" in str(inspect.signature(X.trpo_step))
    # and the mechanism itself: with noise on the action, saturation must rise
    # with sigma; with noise on the cost map it does not
    env = X.Env(32, 0, 100000, X.nominal_spec(speed=(1.2, 1.2)), ("circle",))
    o, e, xr = env.obs()
    th = X.costmap_init(jax.random.PRNGKey(0), env.obs_dim, 16, "diag", 1)
    xs, us = env.ref_traj(1), env.ref_useq(1)
    S, cc = X.costmap_apply(th, o, "diag", 1)
    Pt = jnp.broadcast_to(X.PTt, (32, X.NE, X.NE))
    mu = us[:, 0] + X.ilqr_solve(e, xs, S, cc, Pt, None, None, 6, 0, us)[:, 0]
    sats = []
    for sg in (0.02, 0.10, 0.30):
        a = mu + sg * jax.random.normal(jax.random.PRNGKey(3), mu.shape)
        a = jnp.clip(a, jnp.asarray(X.U_LO), jnp.asarray(X.U_HI))
        sats.append(float(((a[:, 0] <= 1e-9) | (a[:, 0] >= 1 - 1e-9)).mean()))
    assert sats[2] >= sats[0], f"saturation does not rise with sigma: {sats}"


def test_B5_sigma_is_learnable():
    import inspect
    src = inspect.getsource(X.train_ppo)
    assert '"log_sigma": jnp.full' in src, "log_sigma is not in the parameter tree"
    assert 'p["log_sigma"]' in src, "log_sigma never reaches the loss"


# --------------------------------------------------------------------------- #
# C. the adaptive mechanism against arXiv:2605.16015
# --------------------------------------------------------------------------- #
def test_C2_dr_moment_fraction_matches_the_paper():
    mg, arm = X.M_TOT * X.P.g, 0.174
    F, T = A.wrench_dr_magnitudes(0.32)
    assert abs(F / mg - 0.32) < 1e-9
    # paper: 0.001 N.m on a 31.5 g, 28 mm Crazyflie = 0.1156 of weight*arm
    assert abs(T / (mg * arm) - 0.1156) < 2e-3


def test_C3_C4_C5_scenarios_match_the_paper():
    assert A.SCEN_LEVELS["asym"] == (0.0, 0.01, 0.07, 0.11)
    assert A.SCEN_LEVELS["central"] == (0.0, 0.075, 0.15, 0.25)
    assert abs(A.SLUNG_L - 0.174) < 1e-12          # tether = arm length
    assert abs(A.SLUNG_MFRAC - 0.149) < 1e-9
    assert abs(A.SLUNG_PERIOD - 0.8369) < 1e-3
    assert A.H_DEFAULT == 64                       # paper's window
    p = A.rdp_init(jax.random.PRNGKey(0), "GRU")
    assert p["hid"] == (64, 64)                    # paper's width


def test_C6_position_hold_is_a_fixed_point():
    """The RDP trains on position hold, and a hold reference has zero motion.

    Freezing only the clock would leave v_ref at its t=0 value -- 1.5 m/s on a
    circle -- while telling the vehicle to hold.
    """
    env = X.Env(4, 0, 100000, X.nominal_spec(speed=(1.5, 1.5)), ("circle",),
                task="stabilize")
    xr, ur, a = env.ref_now()
    assert float(jnp.abs(xr[:, 3:6]).max()) == 0.0        # v_ref = 0
    assert float(jnp.abs(a).max()) == 0.0                 # a_ref = 0
    assert float(jnp.abs(xr[:, 6:10] - jnp.asarray([1.0, 0, 0, 0])).max()) < 1e-12
    assert abs(float(ur[:, 0].mean()) - X.U_HOVER) < 1e-9
    xs = env.ref_traj(10)
    assert float(jnp.abs(xs[:, -1] - xs[:, 0]).max()) == 0.0
    trk = X.Env(4, 0, 100000, X.nominal_spec(speed=(1.5, 1.5)), ("circle",),
                task="track")
    assert float(jnp.linalg.norm(trk.ref_now()[0][:, 3:6], axis=-1).mean()) > 1.0


def test_E3_task_field_is_read():
    import inspect
    src = inspect.getsource(X)
    assert 'cfg.task == "stabilize"' in src, "EnvCfg.task is still a dead field"


def test_B4_normaliser_is_floored_and_clipped():
    """A near-constant observation channel must not blow up the normaliser.

    Several channels are near-constant within a batch -- a frozen preview under
    position hold, an integral that has not moved -- and an unfloored
    1/sqrt(var) amplifies the first sample that does move by ~1e4.  Measured,
    that took the model-free arm to NaN within three iterations.
    """
    th = {"obs_norm": {"mu": jnp.zeros(3), "var": jnp.asarray([1.0, 0.0, 1e-12]),
                       "count": jnp.asarray(100.0)}}
    o = jnp.asarray([[0.0, 5.0, 5.0]])
    n = X.normalise_obs(th, o)
    assert bool(jnp.all(jnp.isfinite(n)))
    assert float(jnp.abs(n).max()) <= X.OBS_CLIP + 1e-9
    assert X.OBS_VAR_FLOOR > 0.0


def test_B4_model_free_arm_is_normalised_too():
    """§8.3's exploration comparison must not be measuring the normalisation."""
    p = X.mlp_policy_init(jax.random.PRNGKey(0), X.OBS_DIM, 8)
    assert "obs_norm" in p and "net" in p


def test_C5_smoothing_uses_a_fixed_shape_ring_buffer():
    """The rolling mean must not change shape as its buffer fills.

    A growing Python list makes ``jnp.stack`` see a new shape on every step,
    and eager JAX compiles per shape.  Measured, that cost 68 ms per step while
    the buffer filled against 0.29 ms once saturated -- and since
    ``solve_latency_ms`` probes only T = 30 steps from a fresh episode, the
    entire probe landed inside the warm-up and reported the adaptive controller
    at 45 ms median / 112 ms p95 instead of 4.2 / 4.8.  That is a measurement
    artefact reported as a design property, on the one number that decides
    deployability.
    """
    import inspect
    src = inspect.getsource(A.AdaptEnv.attach)
    # match the call, not the prose: the comment above the fix names jnp.stack
    assert 'jnp.stack(st[' not in src, \
        "smoothing is stacking a variable-length list again"
    assert '.append(' not in src.split("def est(")[1].split("return y")[0], \
        "the estimator is growing a Python list again"
    assert "jnp.zeros((n_s,)" in src, "ring buffer is not preallocated at a fixed shape"


def test_C5_ring_buffer_matches_the_growing_list_exactly():
    """The fix must be a performance change only, not a semantics change."""
    n_s = 32

    @jax.jit
    def _roll(buf, y, k):
        buf = jnp.concatenate([buf[1:], y[None]], 0)
        keep = (jnp.arange(buf.shape[0]) >= buf.shape[0] - k)[:, None, None]
        return buf, jnp.sum(jnp.where(keep, buf, 0.0), 0) / k

    rng = np.random.default_rng(0)
    ys = [jnp.asarray(rng.normal(size=(3, 6))) for _ in range(80)]

    ref, lst = [], []
    for y in ys:                                   # the original semantics
        lst.append(y)
        if len(lst) > n_s:
            lst.pop(0)
        ref.append(np.asarray(jnp.mean(jnp.stack(lst), 0)))

    got, buf, cnt = [], None, 0
    for y in ys:
        if buf is None:
            buf = jnp.zeros((n_s,) + y.shape, y.dtype)
        cnt = min(cnt + 1, n_s)
        buf, m = _roll(buf, y, jnp.asarray(cnt))
        got.append(np.asarray(m))

    err = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
    assert err == 0.0, f"ring buffer changed the smoothing: max err {err:e}"


def test_D8_hot_path_emits_no_concatenate():
    """fc/step_c must not assemble their output with jnp.concatenate.

    On a CUDA backend XLA aborts compiling a concatenate whose operands are
    3 wide -- `3 % 4 != 0` against the 4-wide f64 tile it picks -- which killed
    every notebook on an RTX 4080.  No --xla_cpu_* flag reaches it, because the
    computation is on the GPU.  Assembling by scatter avoids the emitter.
    """
    import ast
    import inspect
    import textwrap
    for name in ("fc", "step_c"):
        fn = getattr(X, name)
        fn = getattr(fn, "__wrapped__", fn)          # unwrap jax.jit
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        # ast drops comments, and this strips the docstring, so only real code
        # is inspected -- the comments here deliberately mention concatenate
        code = ast.dump(tree)
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "concatenate"]
        assert not calls, (
            f"{name} assembles with concatenate again; this crashes the CUDA "
            f"backend (see docs/TROUBLESHOOTING.md)")
        assert code  # keep the parse meaningful


def test_D8_assemble_matches_concatenate():
    """_assemble must be exactly jnp.concatenate, or the rewrite is a bug."""
    rng = np.random.default_rng(3)
    for shp in ((8,), (4, 5), (1,)):
        a = jnp.asarray(rng.normal(size=shp + (3,)))
        b = jnp.asarray(rng.normal(size=shp + (3,)))
        c = jnp.asarray(rng.normal(size=shp + (4,)))
        want = jnp.concatenate([a, b, c], -1)
        got = X._assemble(shp, 10, [(0, 3, a), (3, 6, b), (6, 10, c)],
                          jnp.result_type(a, b, c))
        assert got.shape == want.shape
        assert float(jnp.abs(got - want).max()) == 0.0


def test_D8_fc_and_step_c_are_well_formed():
    """fc/step_c must stay finite and keep their invariants on random states."""
    rng = np.random.default_rng(7)
    x = jnp.asarray(np.c_[rng.normal(size=(8, 6)), rng.normal(size=(8, 4)),
                          rng.normal(size=(8, 3)),
                          rng.uniform(X.P.Om_min, X.P.Om_max, size=(8, 4))])
    x = x.at[..., X.CQ].set(X.qnorm(x[..., X.CQ]))
    u = jnp.asarray(rng.uniform(0.1, 0.9, size=(8, 4)))
    d = jnp.asarray(rng.normal(size=(8, 6)) * 0.1)
    f, sc = X.fc(x, u, d), X.step_c(x, u, d)
    assert f.shape == (8, X.NX) and sc.shape == (8, X.NX)
    assert bool(jnp.all(jnp.isfinite(f))) and bool(jnp.all(jnp.isfinite(sc)))
    # step_c must leave a unit quaternion and physical rotor speeds
    n = jnp.linalg.norm(sc[..., X.CQ], axis=-1)
    assert float(jnp.abs(n - 1.0).max()) < 1e-12
    assert float(sc[..., X.CO].min()) >= 0.0
    assert float(sc[..., X.CO].max()) <= X.P.Om_max + 1e-9
    # the gradient must survive the rotor clamp and the sqrt in the allocator
    g = jax.jacobian(lambda uu: X.step_c(x, uu))(u)
    assert bool(jnp.all(jnp.isfinite(g))), "step_c has a non-finite Jacobian in u"
