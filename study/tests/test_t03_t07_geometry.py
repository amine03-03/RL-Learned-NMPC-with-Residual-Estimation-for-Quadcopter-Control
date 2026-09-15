"""T-3, T-4, T-6, T-7 -- error coordinates, path derivatives, error dynamics
and the stage Jacobians."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import x500_core_jax as X


def _rand_q(rng, max_deg):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    th = rng.uniform(0, np.deg2rad(max_deg))
    return np.r_[np.cos(th / 2), ax * np.sin(th / 2)]


def test_T3_error_round_trip():
    """T-3: err(e_to_state(err(x,xr),xr),xr) for 1000 attitudes to 150 deg."""
    rng = np.random.default_rng(3)
    xs = np.array([np.r_[rng.normal(size=6), _rand_q(rng, 150)] for _ in range(1000)])
    xrs = np.array([np.r_[rng.normal(size=6), _rand_q(rng, 150)] for _ in range(1000)])
    x, xr = jnp.asarray(xs), jnp.asarray(xrs)
    e = X.err(x, xr)
    assert float(jnp.abs(X.err(X.e_to_state(e, xr), xr) - e).max()) < 1e-9


def test_T3_shortcut_is_not_used():
    """The shortcut n = ||delta|| is wrong by -2.70 deg at 60 deg; e_to_state
    must not be using it."""
    for deg in (60.0, 120.0):
        th = np.deg2rad(deg)
        e = jnp.zeros((1, 9)).at[0, 6].set(2 * np.sin(th / 2))
        xr = jnp.asarray([[0., 0, 1.5, 0, 0, 0, 1., 0, 0, 0]])
        q = X.e_to_state(e, xr)[0, 6:10]
        ang = 2 * np.arcsin(min(1.0, float(jnp.linalg.norm(q[1:]))))
        assert abs(np.rad2deg(ang) - deg) < 1e-6, f"{deg}: got {np.rad2deg(ang)}"


def test_T3_gradient_is_finite_at_zero_error():
    """N-1: the spec's form of (4.2) divides by ||delta||; its gradient at zero
    error is NaN -- and zero error is exactly where the iLQR linearises."""
    xr = jnp.asarray([[0., 0, 1.5, 0, 0, 0, 1., 0, 0, 0]])
    g = jax.jacobian(lambda e: X.e_to_state(e, xr))(jnp.zeros((1, 9)))
    assert bool(jnp.all(jnp.isfinite(g))), "e_to_state has a non-finite Jacobian at e=0"


@pytest.mark.parametrize("kind", X.PATHS_ALL)
def test_T4_path_derivatives(kind):
    """T-4: v_ref, a_ref against a central difference, all six kinds."""
    i = X.PATH_IDX[kind]
    ep = dict(kind=jnp.asarray([i]), c=jnp.asarray([[0., 0., 1.5]]),
              R=jnp.asarray([1.3]), phi0=jnp.asarray([0.7]),
              omega=jnp.asarray([1.1]), spd=jnp.asarray([1.4]),
              delta=jnp.asarray([[0.3, -0.2, 0.1]]))
    h = 1e-6
    worst_v = worst_a = 0.0
    for t in np.linspace(0.0, 6.0, 40):
        p0, v0, a0 = X._ref_pva(ep, jnp.asarray([t]))
        pp, vp, _ = X._ref_pva(ep, jnp.asarray([t + h]))
        pm, vm, _ = X._ref_pva(ep, jnp.asarray([t - h]))
        sc_v = max(float(jnp.abs(v0).max()), 1.0)
        sc_a = max(float(jnp.abs(a0).max()), 1.0)
        worst_v = max(worst_v, float(jnp.abs((pp - pm) / (2 * h) - v0).max()) / sc_v)
        worst_a = max(worst_a, float(jnp.abs((vp - vm) / (2 * h) - a0).max()) / sc_a)
    assert worst_v < 1e-6, f"{kind}: v_ref vs fd {worst_v}"
    assert worst_a < 1e-6, f"{kind}: a_ref vs fd {worst_a}"


def test_T4_kappa_table():
    """T-4: the §4.2 peak constants, recomputed by dense sampling."""
    want_v = (0., 0., 1.0, 1.4142136, 1.0307764, 1.4160340)
    want_a = (0., 0., 1.0, 2.1250000, 1.0, 9.0778770)
    for i, k in enumerate(X.PATHS_ALL):
        assert abs(X.KAPPA_V[i] - want_v[i]) < 1e-6, k
        assert abs(X.KAPPA_A[i] - want_a[i]) < 1e-6, k
    # fig8: kappa_a^2 = 289/64 exactly (proved in §4.2 by maximising 17u-16u^2).
    # KAPPA_A is obtained by sampling 2e5 points, whose residual near a smooth
    # maximum is O(grid^2) ~ 5e-10 in kappa; 1e-8 in kappa^2 is the resolution
    # limit of the estimator, not a loosened requirement.
    assert abs(X.KAPPA_A[3] ** 2 - 289 / 64) < 1e-8


def test_T4_feasibility_cap_binds():
    """(4.6): uncapped, the superellipse at R=0.5, spd=1.5 demands 40.85 m/s^2
    against 13.35 available.  Capped, no path may exceed alpha*a_lat_max."""
    R, spd = 0.5, 1.5
    w_un = spd / max(R, 0.3)
    assert abs(R * w_un ** 2 * X.KAPPA_A[5] - 40.85) < 0.1
    for kind in X.PATHS_ALL:
        w = float(X.path_omega(kind, R, spd))
        _, pa = X.path_demand(kind, R, w)
        assert pa <= X.P.alpha_feas * X.A_LAT_MAX + 1e-6, (kind, pa)


def test_T6_edyn_matches_step_c_then_err():
    """T-6: edyn == err(step_c(e_to_state(.)), xr_next) to 1e-9."""
    rng = np.random.default_rng(6)
    B = 8
    e = jnp.asarray(rng.normal(size=(B, 9)) * 0.2)
    xr = jnp.asarray(np.c_[rng.normal(size=(B, 6)), np.tile([1., 0, 0, 0], (B, 1))])
    xr = jnp.concatenate([xr[:, :6], X.qnorm(xr[:, 6:10])], -1)
    xrn = xr
    du = jnp.asarray(rng.normal(size=(B, 4)) * 0.05)
    ur = jnp.zeros((B, 4)).at[:, 0].set(X.U_HOVER)
    direct = X.err(X.step_c(X.e_to_state(e, xr), ur + du), xrn)
    assert float(jnp.abs(X.edyn(e, du, xr, xrn, None, None, ur) - direct).max()) < 1e-9


def test_T6_moving_reference_is_not_frozen():
    """Freezing the reference is a bug, not a simplification: the two rollouts
    must differ on a moving reference."""
    env = X.Env(4, 0, 10000, X.nominal_spec(speed=(1.5, 1.5)), ("circle",))
    xr_seq = env.ref_traj(20)
    frozen = jnp.broadcast_to(xr_seq[:, :1], xr_seq.shape)
    assert float(jnp.abs(xr_seq - frozen).max()) > 0.1, \
        "the preview is not moving -- ref_traj is returning a frozen reference"
    o, e, xr = env.obs()
    ur = env.ref_useq(20)
    du = jnp.zeros((4, 20, 4))
    a = X.rollout_err(e, du, xr_seq, ur)
    b = X.rollout_err(e, du, frozen, ur)
    assert float(jnp.abs(a[:, -1] - b[:, -1]).max()) > 0.1


@pytest.mark.parametrize("at_boundary", [False, True])
def test_T7_lin_traj_vs_finite_difference(at_boundary):
    """T-7: A, B against finite differences, **including at the clamp boundary**.

    N-2: `_clamp` passes the derivative through *on* the boundary, so the
    reference there is a one-sided difference taken into the feasible interior.
    A central difference across the kink would return 1/2 and would fail a
    correct implementation.
    """
    rng = np.random.default_rng(7)
    B, k = 4, 0
    e0 = jnp.asarray(rng.normal(size=(B, 9)) * 0.2)
    xr_seq = jnp.broadcast_to(jnp.asarray([0., 0, 1.5, 0, 0, 0, 1., 0, 0, 0]), (B, 3, 10))
    ur = jnp.broadcast_to(jnp.asarray([X.U_HOVER, 0., 0., 0.]), (B, 2, 4))
    du = (jnp.zeros((B, 2, 4)).at[:, :, 0].set(1.0 - X.U_HOVER) if at_boundary
          else jnp.zeros((B, 2, 4)) + 0.05)
    A, Bm = X.lin_traj(e0, du, xr_seq, None, None, ur)
    f = lambda ee, dd: X.edyn(ee, dd, xr_seq[:, k], xr_seq[:, k + 1], None, None, ur[:, k])
    h = 1e-6
    Afd = np.zeros((B, 9, 9))
    Bfd = np.zeros((B, 9, 4))
    for j in range(9):
        Afd[:, :, j] = (f(e0.at[:, j].add(h), du[:, k])
                        - f(e0.at[:, j].add(-h), du[:, k])) / (2 * h)
    hi = np.asarray(X.U_HI) - np.asarray(ur)[0, k]
    for j in range(4):
        on_edge = bool(np.asarray(du)[0, k, j] >= hi[j] - 1e-12)
        if on_edge:
            Bfd[:, :, j] = (f(e0, du[:, k]) - f(e0, du[:, k].at[:, j].add(-h))) / h
        else:
            Bfd[:, :, j] = (f(e0, du[:, k].at[:, j].add(h))
                            - f(e0, du[:, k].at[:, j].add(-h))) / (2 * h)
    assert np.abs(np.asarray(A[:, k]) - Afd).max() < 1e-5
    assert np.abs(np.asarray(Bm[:, k]) - Bfd).max() < 1e-5
