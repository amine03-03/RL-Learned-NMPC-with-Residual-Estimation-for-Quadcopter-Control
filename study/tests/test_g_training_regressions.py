"""Regression tests for the G-series findings of the medium-run debug.

Every one of these was silent: the pipeline ran to completion, printed tables,
and the tables were a property of the initialiser or of a measurement bound
rather than of the algorithm under test.  The G-series is documented in
docs/AUDIT.md.
"""
import inspect

import numpy as np
import jax
import jax.numpy as jnp

import adaptive_core_jax as A
import x500_core_jax as X


def _core_src():
    return open(X.__file__).read()


def _func_src(mod_src, name):
    """Source of one top-level function, by text.

    ``inspect.getsource`` resolves ``co_firstlineno`` against the file on disk,
    so it silently returns a DIFFERENT function's block if the module was edited
    after import -- which is exactly what a source-inspection test must not be
    sensitive to.
    """
    return mod_src.split(f"\ndef {name}(")[1].split("\ndef ")[0]


def _code_only(src):
    """Drop comment lines.

    These assertions are about CODE.  The fixes carry long comments that quote
    the broken lines verbatim -- which is the point of them -- so matching the
    raw source would make every such comment fail its own test.
    """
    return "\n".join(ln for ln in src.split("\n")
                      if not ln.lstrip().startswith("#"))


def _cfg(**kw):
    c = dict(N=1, rep="diag", n_iter=4, n_diff=1, hid=16, minib=4, epochs=2,
             sigma=0.05, algo="ppo", mpve=False, lam=0.95, dist_label="test")
    c.update(kw)
    return c


# --------------------------------------------------------------------------- #
# G1  the trust region must not be a one-way ratchet on the learning rate
# --------------------------------------------------------------------------- #
def test_G1_some_policy_step_actually_lands():
    """The PPO branch must accept at least one step.

    Before the fix a KL trip set ``stop = True``, abandoning every remaining
    minibatch *and* epoch, and the one step it had taken was reverted -- actor,
    critic and log_sigma together.  Measured on the shipped code: 8
    ``apply_updates`` calls over 8 iterations (against 256 intended), all
    reverted, with ``sigma`` and ``entropy`` byte-identical down every column.
    Every AC-MPC row in the study was therefore the initialiser.
    """
    env = X.Env(8, 0, 120, X.nominal_spec(speed=(1.0, 1.2)), ("circle",))
    _, _, log = X.train_ppo(None, _cfg(), seed=0, iters=4, T_rollout=8,
                            env=env, verbose=False)
    assert "landed" in log, "train_ppo must report how many steps were accepted"
    assert int(log["landed"].sum()) > 0, (
        "not one policy step was accepted; the actor is its initialisation")


def test_G1_lr_is_not_a_one_way_ratchet():
    """``lr`` must be able to recover, and the growth branch must be reachable.

    ``if kl_seen and kl_seen < 0.5 * kl_target`` is False when the first
    minibatch trips, because ``kl_seen`` is still the float 0.0 -- which Python
    reads as falsy.  So the decay fired every iteration and the growth never
    did: measured, 3e-4 -> 1e-7 (``lr_min``) in 12 iterations, monotonically, at
    every scale.  That is why ``full`` cannot fix the AC-MPC rows.
    """
    src = _code_only(_func_src(_core_src(), "train_ppo"))
    assert "if kl_seen and kl_seen <" not in src, (
        "the lr growth branch is still gated on a float that is 0.0 whenever no "
        "step landed")
    assert "if n_land and kl_seen <" in src
    # and the mechanism: the region is a backtracking search inside the
    # minibatch, so a trip costs that minibatch and not the whole iteration
    assert "kl_backtracks" in X.PPO_DEFAULTS
    assert X.PPO_DEFAULTS["kl_backtracks"] >= 1
    # the old control flow hinged on a `stop` flag initialised before the epoch
    # loop and set on the first trip
    assert "stop = False" not in src and "stop = True" not in src, (
        "a KL trip still abandons the remaining minibatches and epochs")
    assert "trips >= c[\"kl_give_up\"]" in src, (
        "there is no bounded per-iteration trip budget")


def test_G1_critic_step_survives_a_rejected_actor_step():
    src = _code_only(_func_src(_core_src(), "train_ppo"))
    assert 'dict(base, critic=trial["critic"])' in src, (
        "the critic update is still rolled back with the actor; the critic is a "
        "supervised regression on the returns and has nothing to do with the "
        "policy trust region")


# --------------------------------------------------------------------------- #
# G2  leaving the 3 m ball must not be the most profitable action available
# --------------------------------------------------------------------------- #
def test_G2_leaving_the_ball_is_priced():
    """`far` must reach the reward, not only ``done``.

    Every reward of (5.4) is strictly negative, so cutting the value bootstrap
    at a free termination makes the return of a terminated episode ``r_k``
    instead of ``r_k + gamma V``.  Measured with a consistent critic at
    r = -3/step and gamma = 0.99: advantage +297.0 for the escape, against a
    crash penalty of 5.0.  PPO was being trained to fly out of the ball, which
    is exactly what Notebook 7 films (0.09 rad of a 6.28 rad lap, RMSE 49 m).
    """
    src = _code_only(
        _core_src().split("def _step_jit(")[1].split("\ndef ")[0])
    assert "fail = (bad | spin | far)" in src
    assert "d_u, fail)" in src and "du, om, d_u, fail)" in src


def test_G2_value_mask_bootstraps_through_a_respawn():
    env = X.Env(8, 0, 4, X.nominal_spec(speed=(1.0, 1.2)), ("circle",))
    assert env.cfg.term == "bootstrap"
    o, e, xr = env.obs()
    _, uref, _ = env.ref_now()
    for _ in range(6):                                # long enough to respawn
        r, done, info = env.step(uref)
        assert "done_value" in info
        assert float(jnp.max(info["done_value"])) == 0.0, (
            "term='bootstrap' must not cut the value target at a respawn")
    env2 = X.Env(8, 0, 4, X.nominal_spec(speed=(1.0, 1.2)), ("circle",),
                 term="cut")
    assert env2.cfg.term == "cut"                     # the old behaviour is kept


def test_G2_escape_advantage_is_not_positive_under_bootstrapping():
    T, B, gamma, lam = 30, 1, 0.99, 0.95
    rew = jnp.full((T, B), -3.0)
    val = jnp.full((T, B), -300.0)                    # a consistent critic
    cut = X.gae(rew, val, val[-1], jnp.zeros((T, B)).at[15].set(1.0), gamma, lam)
    boot = X.gae(rew, val, val[-1], jnp.zeros((T, B)), gamma, lam)
    assert float(cut[15, 0]) > 100.0, "the pathology must still be reproducible"
    assert abs(float(boot[15, 0])) < 1.0, (
        "bootstrapping through the boundary must remove the free lunch")


# --------------------------------------------------------------------------- #
# G3  the running observation statistics are not parameters
# --------------------------------------------------------------------------- #
def test_G3_obs_norm_is_frozen_under_the_optimiser():
    mod = _core_src()
    src = _code_only(_func_src(mod, "train_ppo"))
    assert "_freeze_norm" in src, "obs_norm still receives an optimiser update"
    src_t = _code_only(_func_src(mod, "trpo_step"))
    assert 'k != "obs_norm"' in src_t, (
        "trpo_step still flattens obs_norm into the natural-gradient direction")


def test_G3_trpo_leaves_the_normaliser_alone():
    env = X.Env(8, 0, 120, X.nominal_spec(speed=(1.0, 1.2)), ("circle",))
    th = X.costmap_init(jax.random.PRNGKey(0), env.obs_dim, 16, "diag", 1)
    o, e, xr = env.obs()
    xs, us = env.ref_traj(1), env.ref_useq(1)
    ls = jnp.full((X.NU,), np.log(0.05))
    mu = us[:, 0] + X.ilqr_solve(e, xs, *X.costmap_apply(th, o, "diag", 1),
                                 jnp.broadcast_to(X.PTt, (8, X.NE, X.NE)),
                                 None, None, 4, 0, us)[:, 0]
    a = mu + 0.05 * jax.random.normal(jax.random.PRNGKey(1), mu.shape)
    lp = X._logp(a, mu, ls)
    act_mu = lambda p, ob: (us[:, 0] + X.ilqr_solve(
        e, xs, *X.costmap_apply(p, ob, "diag", 1),
        jnp.broadcast_to(X.PTt, (8, X.NE, X.NE)), None, None, 4, 0, us)[:, 0])
    new, _, _ = X.trpo_step(th, ls, act_mu, o, a, lp,
                            jax.random.normal(jax.random.PRNGKey(2), (8,)))
    for k in ("mu", "var", "count"):
        assert np.allclose(np.asarray(new["obs_norm"][k]),
                           np.asarray(th["obs_norm"][k])), \
            f"trpo_step moved obs_norm[{k!r}]"


# --------------------------------------------------------------------------- #
# G4/G5  AdaptEnv: per-vehicle disturbance and per-vehicle window
# --------------------------------------------------------------------------- #
def test_G4_wrench_resample_is_per_vehicle():
    env = A.AdaptEnv(6, 0, 200, X.nominal_spec(speed=(1.0, 1.2)), "central", 0.0,
                     wrench_dr=0.10, paths=("circle",), H=8)
    w0 = np.asarray(env._dr_wrench).copy()
    mask = jnp.asarray([True, False, False, True, False, False])
    env._sample_wrench_dr(mask=mask)
    w1 = np.asarray(env._dr_wrench)
    moved = np.abs(w1 - w0).max(1) > 0
    assert list(moved) == [True, False, False, True, False, False], (
        "a respawn on one vehicle still redraws the wrench for the whole batch; "
        "at n_env=128 that fires on ~85 % of steps and turns the per-episode "
        "wrench into 50 Hz white noise")


def test_G4_window_is_cleared_per_vehicle_on_respawn():
    env = A.AdaptEnv(4, 0, 200, X.nominal_spec(speed=(1.0, 1.2)), "central", 0.0,
                     paths=("circle",), H=6)
    for _ in range(8):
        env.push_frame()
    assert np.asarray(env.ready()).all()
    env._clear_windows(jnp.asarray([True, False, True, False]))
    assert list(np.asarray(env.ready())) == [False, True, False, True], (
        "deployment windows still straddle an episode boundary, while "
        "make_windows refuses exactly that for the training windows (§9.6)")
    assert np.abs(env._buf[:, 0, :]).max() == 0.0


def test_G5_estimator_runs_once_per_control_step():
    env = A.AdaptEnv(4, 0, 200, X.nominal_spec(speed=(1.0, 1.2)), "central", 0.0,
                     paths=("circle",), H=4, oracle=True)
    calls = {"n": 0}

    def fake(e):
        calls["n"] += 1
        if e._pred is not None and e._pred_ver == e._buf_ver:
            return e._pred
        y = jnp.zeros((e.n, 6))
        e._pred, e._pred_ver = y, e._buf_ver
        return y

    env.estimator = fake
    env.obs(); env.dmod(); env.d_channel()
    assert calls["n"] == 3                      # reached three times ...
    assert env._pred_ver == env._buf_ver        # ... but evaluated once
    env.push_frame()
    assert env._pred_ver != env._buf_ver, "a new frame must invalidate the cache"


# --------------------------------------------------------------------------- #
# G7  the LLTC fit must score the object the controller evaluates
# --------------------------------------------------------------------------- #
def test_G7_lltc_acceptance_gate_is_not_a_tautology():
    """Only ONE of G7's three changes survived measurement.

    G7 changed three things in nb2 and all three were argued rather than
    measured.  Bisected afterwards at X500_SCALE=medium, LLTC N=1 on S1:

        original                                   0.4134 m
        + decades + circular gate      (G7)        1.2281 m
        + decades, honest gate         (G7b)       1.2708 m
        + one shell, honest gate       (G7c)       1.0640 m
        + e0 contraction               (G7d)       0.4566 m

    so the e0 -> e1 contraction cost 2.33x and the decade sampling 1.19x.
    Both were reverted.  What is pinned here is the one change that stands:
    the acceptance gate.  A 95th-percentile cut reports 0.95 for ANY input,
    which is why that column read 0.9492 in every row of the terminal-weight
    sweep while `reach_m` moved 1.07 -> 0.29 m.  It cannot affect the
    controller, only the honesty of the diagnostic.
    """
    import os
    nb = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "notebooks", "nb2_lltc.py")
    src = open(nb).read()
    assert "np.percentile(V1[np.isfinite(V1)], 95)" not in src, (
        "the acceptance gate is a 95th-percentile cut again, i.e. 0.95 for any "
        "input -- a tautology printed as a diagnostic")
    assert "keep = np.isfinite(V1) & (V1 > 0.0)" in src
    # and the two reverts stay reverted, with the measurement behind them
    assert 'jnp.einsum("bi,bij,bj->b", ee, P, ee)' in src, (
        "the fit contracts with e1 again; measured, that costs 2.33x because "
        "the fit's e1 is from the N=10 hand-weighted trajectory while the "
        "controller's is from its own N=1 solve -- not the same e1")
    assert "decade = 10.0 **" not in src, (
        "the decade-spread sampling is back; measured, it costs 1.19x because "
        "an MSE on V1 is insensitive to the small-error samples it adds")


# --------------------------------------------------------------------------- #
# G8  the PID tilt limit must be live
# --------------------------------------------------------------------------- #
def test_G8_pid_tilt_limit_is_applied():
    src = _code_only(_func_src(_core_src(), "make_pid_ctrl"))
    assert "a_lim" in src and "ref_attitude(a_lim)" in src, (
        "zb_des is still clamped and then never read")
    assert "(root - r_idle) / (1.0 - r_idle)" in src, (
        "the PID collective still ignores the idle floor of (2.10)")
    ctrl = X.make_pid_ctrl()
    env = X.Env(4, 0, 200, X.nominal_spec(speed=(1.0, 1.2)), ("circle",))
    o, e, xr = env.obs()
    # a 5 m lateral error demands far more tilt than tilt_max
    e_big = e.at[:, 0].set(5.0)
    u, _ = ctrl(o, e_big, xr)
    x = X.e_to_state(e_big, xr)
    zb = X.qzaxis(X.ref_attitude(-jnp.asarray(X.PID_DEFAULT["kp_pos"]) * e_big[:, 0:3]))
    assert np.isfinite(np.asarray(u)).all()
    assert float(jnp.max(u[:, 0])) <= 1.0 and float(jnp.min(u[:, 0])) >= 0.0


# --------------------------------------------------------------------------- #
# G9  RMSE under respawn is bounded, and stats() must say so
# --------------------------------------------------------------------------- #
def test_G9_stats_reports_the_bound():
    env = X.Env(8, 0, 100000, X.nominal_spec(speed=(1.0, 1.2)), ("circle",))
    st = X.stats(X.rollout_eval(env, X.make_lqr_ctrl(), 40))
    for k in ("bound_frac", "bounded"):
        assert k in st, f"stats() must report {k!r} beside rmse"
    assert st["bounded"] is False and st["rmse"] < 0.5


def test_G9_the_bound_caps_rmse_for_a_diverging_controller():
    """With respawn on, |e_p| <= MAX_POS_ERR, so RMSE cannot grow with T.

    This is the whole reason every learned row of the medium run reads ~1.07 m on
    every suite, every path and every disturbance level while Notebook 7 -- the
    only place ``no_respawn`` is set -- reads 45-50 m for the same checkpoints.
    Measured on an untrained cost map (hid=256, ilqr=10): 0.892 m at T=150 and
    0.904 m at T=300 with respawn on, 11.39 m then 41.36 m with it off.

    Driven here by a controller that diverges BY CONSTRUCTION -- full collective,
    so it climbs away from any reference -- because whether a given random
    initialisation diverges is a property of that draw, and this test is about
    the measurement and not about the initialiser.
    """
    def runaway(o, e, xr, uref=None, d=None):
        return jnp.zeros((e.shape[0], X.NU)).at[:, 0].set(1.0), {}
    runaway.name = "full collective"

    out = {}
    for no_res in (False, True):
        for T in (60, 120):
            env = X.Env(8, 900, 100000, X.nominal_spec(speed=(1.0, 1.2)),
                        ("circle",))
            env.no_respawn = no_res
            out[(no_res, T)] = X.stats(X.rollout_eval(env, runaway, T, warmup=10))
    bounded = out[(False, 60)]["rmse"], out[(False, 120)]["rmse"]
    free = out[(True, 60)]["rmse"], out[(True, 120)]["rmse"]
    assert max(bounded) < X.MAX_POS_ERR, (
        f"the bound must cap rmse, got {bounded}")
    assert free[1] > free[0] * 1.5, (
        f"without the bound divergence must show as growth, got {free}")
    assert free[1] > 3.0 * max(bounded), (
        f"respawn-clipped {bounded} is not distinguishable from free {free}; the "
        "measurement trap of §5.10 is not being exercised")
    assert out[(False, 120)]["bound_frac"] > 0.0
    assert out[(False, 120)]["bounded"] is True
