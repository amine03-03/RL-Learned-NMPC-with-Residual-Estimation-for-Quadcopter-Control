"""T-8 .. T-12 -- the two disturbance objects and the interface between them.

T-8 is the single most valuable test here: a residual that is not identically
zero on an undisturbed plant makes every downstream estimate meaningless, and
nothing else would notice.
"""
import numpy as np
import jax.numpy as jnp
import pytest

import adaptive_core_jax as A
import x500_core_jax as X


def test_T8_zero_on_the_nominal_plant():
    """T-8: external_wrench and true_disturbance are **zero** on the nominal
    plant with no payload, wind or wrench."""
    par = X.make_par(4)
    s = X.hover_state(4, par=par)
    u = X.hover_u(4, par=par)
    assert float(jnp.abs(X.true_disturbance(s, u, par)).max()) < 1e-9
    assert float(jnp.abs(X.external_wrench(s, u, par, X.fp(s, u, par))).max()) < 1e-9
    # and the trim really is a trim
    assert float(jnp.abs(X.fp(s, u, par)).max()) < 1e-12


def test_T8_holds_after_flying():
    """Still zero once the vehicle is actually flying a nominal trajectory:
    drag and rotor lag are the only model errors, and at the trim both vanish."""
    env = X.Env(4, 0, 100000, X.nominal_spec(speed=(0.0, 0.0)), ("hover",),
                fixed=dict(e0=[0.0, 0.0, 0.0]))
    ctrl = X.make_lqr_ctrl()
    for _ in range(200):
        o, e, xr = env.obs()
        _, uref, _ = env.ref_now()
        u, _ = ctrl(o, e, xr, uref=uref)
        env.step(u)
    assert float(jnp.abs(env.d_truth()).max()) < 5e-3, \
        "nominal hover shows a nonzero wrench: the truth computation is wrong"


@pytest.mark.parametrize("scen,lvl", [("central", 0.075), ("central", 0.15),
                                      ("central", 0.25), ("asym", 0.04)])
def test_T9_payload_statics(scen, lvl):
    """T-9: external_wrench matches (9.7) within 5 % in **steady hover**.

    ``asym`` is asserted only below the trim limit of N-7: holding the standing
    moment of an arm-tip payload needs J*K_i*I_lim = 0.286 N.m of integrator
    authority, so the largest trimmable fraction is f = 0.0812 and the §6.1 top
    level of 0.11 is not flyable.  That failure is reported as a result in
    Notebook 5, not hidden by loosening this tolerance.
    """
    env = A.AdaptEnv(2, 1, 200000, X.nominal_spec(speed=(0.0, 0.0)), scen, lvl,
                     paths=("hover",), fixed=dict(e0=[0.0, 0.0, 0.0]))
    ctrl = X.make_lqr_ctrl()
    for _ in range(1500):
        o, e, xr = env.obs()
        _, uref, _ = env.ref_now()
        u, _ = ctrl(o, e, xr, uref=uref)
        env.step(u)
    w = np.asarray(env.d_truth()).mean(0)
    F, tau = A.PayloadScenario(scen, lvl).analytic_wrench()
    assert abs(w[2] - F[2]) <= 0.05 * abs(F[2]), f"F_z {w[2]} vs {F[2]}"
    if abs(tau[1]) > 1e-9:
        assert abs(w[4] - tau[1]) <= 0.05 * abs(tau[1]), f"tau_y {w[4]} vs {tau[1]}"


def test_T9_asym_trim_limit_is_where_N7_says():
    """The trim limit itself, asserted so the N-7 argument cannot rot."""
    tau_avail = X.J_NOM[0, 0] * X.P.K_i[0] * X.P.I_lim
    assert abs(tau_avail - 0.28597) < 1e-4
    f_max = tau_avail / (0.174 * X.M_TOT * X.P.g)
    assert abs(f_max - 0.0812) < 1e-3
    assert 0.07 < f_max < 0.11, "the §6.1 asym levels straddle the trim limit"


def test_T10_dmod_vs_d_channel():
    """T-10: for a known wrench [6.478,0,0,0,0,0], dmod[0] = 3.138 and
    d_channel[0] = 6.478.  Feeding the raw wrench to the model is a factor m
    too large and raises no exception."""
    w = jnp.asarray([[6.478, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert abs(float(w[0, 0]) - 6.478) < 1e-6
    assert abs(float(X.wrench_to_dmod(w)[0, 0]) - 3.138) < 1e-3
    w2 = jnp.asarray([[0.0, 0.0, 0.0, 0.3452, 0.0, 0.0]])
    assert abs(float(X.wrench_to_dmod(w2)[0, 3]) - 1.035) < 1e-3
    assert float(jnp.abs(X.wrench_to_dmod(w2, "none")[:, 3:]).max()) == 0.0
    # the §4.4 worked example, recomputed
    assert abs(0.32 * X.M_TOT * X.P.g - 6.478) < 1e-3
    assert abs(0.098 * X.M_TOT * X.P.g * 0.174 - 0.3452) < 1e-3


def test_T11_first_order_rate_model():
    """T-11: (4.13) against a measured plant response to a constant moment.

    It is a *model*, not an identity -- the acceptance is a factor of 2.
    """
    tau = 0.15
    par = X.par_set_wrench(X.make_par(1), jnp.asarray([[0., 0, 0, tau, 0, 0]]))
    s = X.hover_state(1, par=par)
    u = X.hover_u(1, par=par)
    for _ in range(15):                      # ~4 x the 71 ms rate time constant
        s = X.step_p(s, u, par)
    measured = float(s[0, X.SW][0])
    predicted = float(X.wrench_to_dmod(jnp.asarray([[0., 0, 0, tau, 0, 0]]))[0, 3])
    ratio = measured / predicted
    assert 0.5 < ratio < 2.0, f"(4.13) off by {ratio:.2f}x: {measured} vs {predicted}"


def test_T12_mixer_nominal_vs_true():
    """T-12: with an arm-tip payload the nominal mixer must leave the standing
    moment for the rate loop to fight, and the true mixer must not.

    C-6: the specification's acceptance is on steady ||tau_ext||, which is the
    one quantity the mixer choice does **not** change.  At equilibrium (4.11)
    reduces to ``tau_ext = -(Mtau_nom - Mtau_true) f = -m_p g (0.174,-0.174,0)``,
    a function of the CG shift and the total thrust alone -- the allocator does
    not appear, because (4.11) references the *nominal* geometry by
    construction, which is exactly what makes a payload visible at all.
    Measured, ||tau_ext|| differs by 2-8 % between the two mixers, never 5x.

    What the mixer changes, by an unbounded factor, is the rate-loop effort: a
    true mixer allocates the off-centre load away before it is felt, so
    ``tau_cmd`` and ``I_om`` are identically zero.  That is the design decision
    §3.3 is about, so that is what is asserted here.  ``tau_ext`` is measured
    too and reported, so the claim above stays falsifiable.
    """
    lvl = 0.07               # below the N-7 trim limit, so both cases have a
    out = {}                 # genuine steady state to compare
    for mixer in ("nominal", "true"):
        env = A.AdaptEnv(2, 1, 200000, X.nominal_spec(speed=(0.0, 0.0)), "asym",
                         lvl, paths=("hover",), mixer=mixer,
                         fixed=dict(e0=[0.0, 0.0, 0.0]))
        ctrl = X.make_lqr_ctrl()
        eff, wr = [], []
        for k in range(600):
            o, e, xr = env.obs()
            _, uref, _ = env.ref_now()
            u, _ = ctrl(o, e, xr, uref=uref)
            env.step(u)
            if k >= 500:
                _, _, tau_cmd = X.inner_loop(env.state, u, env.par)
                eff.append(float(jnp.linalg.norm(jnp.mean(tau_cmd, 0))))
                wr.append(float(jnp.linalg.norm(jnp.mean(env.d_truth(), 0)[3:6])))
        out[mixer] = (float(np.mean(eff)), float(np.mean(wr)))

    eff_nom, wr_nom = out["nominal"]
    eff_true, wr_true = out["true"]
    assert eff_nom > 0.05, f"nominal mixer shows no rate-loop effort: {eff_nom}"
    assert eff_nom >= 5.0 * max(eff_true, 1e-6), (
        f"rate-loop effort ||tau_cmd||: nominal={eff_nom:.4f} vs true={eff_true:.4f} "
        "-- the true mixer is not absorbing the payload")
    # and the thing the spec asserts on, which does not discriminate:
    assert abs(wr_nom - wr_true) < 0.25 * max(wr_nom, wr_true), (
        f"||tau_ext|| nominal={wr_nom:.4f} true={wr_true:.4f}: C-6 predicts these "
        "agree to within a few per cent; if they no longer do, external_wrench "
        "has stopped referencing the nominal allocation")


def test_T12_payload_is_visible_in_the_wrench_at_all():
    """The complement of T-12: whichever mixer flies, the standing moment must
    appear in the recorded truth, or there is nothing for the RDP to learn.

    Checked on **direction and order of magnitude**, not against (9.7), because
    this is an instantaneous untrimmed state: there ``omega_dot != 0`` and (4.11)
    carries ``J_nom omega_dot = (J_nom/J_true) M_tau_true f``, i.e. the analytic
    value scaled by the inertia ratio the payload introduces.  The exact match
    with (9.7) belongs in steady hover and is T-9's job.
    """
    lvl = 0.07
    par = X.make_par(2, payload_m=lvl * X.M_TOT, payload_r=[0.174, 0.174, -0.05])
    s = X.hover_state(2, par=par)
    u = X.hover_u(2, par=par)
    tau = np.asarray(X.external_wrench(s, u, par, X.fp(s, u, par)))[0, 3:6]
    want = A.PayloadScenario("asym", lvl).analytic_wrench()[1]
    cos = float(tau @ want / (np.linalg.norm(tau) * np.linalg.norm(want)))
    assert cos > 0.999, f"standing moment points the wrong way: {tau} vs {want}"
    # Exact prediction at this untrimmed instant: tau_ext = J_nom omega_dot with
    # omega_dot = J_true^-1 (Mtau_true f).  J_true is a FULL matrix -- an
    # off-axis payload creates products of inertia -- so the scaling is
    # J_nom @ inv(J_true), not the ratio of the xx entries.
    J_true = np.asarray(par["J"])[0]
    f = np.asarray(par["Mtau_w2"])[0] @ np.asarray(s[0, X.SO]) ** 2
    predicted = X.J_NOM @ np.linalg.solve(J_true, f)
    assert np.abs(tau - predicted).max() < 1e-9, f"{tau} vs {predicted}"
    assert np.linalg.norm(tau) > 0.5 * np.linalg.norm(want), \
        "the payload is barely visible in the wrench"


def test_attached_estimator_reaches_the_observation():
    """An attached RDP must change the observation channel.

    Without this the oracle channel keeps returning ``d_truth`` and every
    "RDP" row is silently an oracle row -- ground truth reaching the online
    controller, which §9.1 forbids.  The failure is invisible in any metric:
    the numbers just come out better than they should.
    """
    import jax
    import adaptive_core_jax as Ad

    def build():
        env = Ad.AdaptEnv(4, 3, 100000, X.nominal_spec(speed=(1.0, 1.0)), "asym",
                          0.04, oracle=True, paths=("circle",), H=8)
        ctrl = X.make_lqr_ctrl()
        for _ in range(12):
            o, e, xr = env.obs()
            _, uref, _ = env.ref_now()
            u, _ = ctrl(o, e, xr, uref=uref)
            env.step(u)
        return env

    truth_env = build()
    o_truth = np.asarray(truth_env.obs()[0])

    rdp_env = build()
    p = Ad.rdp_init(jax.random.PRNGKey(0), "GRU", H=8, hid=(8, 4))
    scales = {"mu": jnp.zeros(Ad.FRAME_DIM), "sd": jnp.ones(Ad.FRAME_DIM),
              "out_sd": jnp.ones(6)}
    rdp_env.attach(p, scales)
    o_rdp = np.asarray(rdp_env.obs()[0])

    assert o_truth.shape[1] == X.OBS_DIM + 6, "the oracle channel is not present"
    assert np.abs(o_truth[:, :X.OBS_DIM] - o_rdp[:, :X.OBS_DIM]).max() < 1e-9, \
        "attaching an estimator changed the base observation, which it must not"
    assert np.abs(o_truth[:, X.OBS_DIM:] - o_rdp[:, X.OBS_DIM:]).max() > 1e-6, \
        "the observation still carries ground truth after attaching an RDP"
    # and d_truth must keep returning the truth for the logger
    assert np.abs(np.asarray(rdp_env.d_truth())
                  - np.asarray(truth_env.d_truth())).max() < 1e-9
