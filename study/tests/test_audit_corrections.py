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
def test_D3_moment_gain_matches_the_plant():
    """moment_gain(t) must reproduce the measured closed-loop response."""
    assert abs(X.moment_gain(0.3)[0] - 0.939) < 0.02
    assert abs(X.moment_gain(8.0)[0] - 0.101) < 0.02     # NOT 1.0 -- (4.13) is a
    assert X.moment_gain(30.0)[0] < 0.01                 # short-transient model
    w = jnp.asarray([[0.0, 0, 0, 0.3452, 0, 0]])
    fo = float(X.wrench_to_dmod(w, "first_order")[0, 3])
    cl = float(X.wrench_to_dmod(w, "closed_loop")[0, 3])
    assert abs(fo - 1.0347) < 1e-3
    assert abs(cl / fo - X.moment_gain(X.DMOD_SETTLE_S)[0]) < 1e-9
    assert float(X.wrench_to_dmod(w, "none")[0, 3]) == 0.0


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
