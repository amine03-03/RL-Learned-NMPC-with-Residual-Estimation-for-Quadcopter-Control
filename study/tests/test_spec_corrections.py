"""The five specification statements that do not survive checking, plus the two
measured notes, each asserted against the measurement that refutes it.

These are not tests of the implementation; they are tests of the *argument* in
docs/CORRECTIONS.md.  If one of them starts failing, the correction is wrong and
the document must change with it.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.linalg import expm, solve_discrete_are

import adaptive_core_jax as A
import study_moderate as M
import x500_core_jax as X

XI = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def test_C1_Ac_sign_and_the_instability_it_causes():
    """C-1: (5.6)'s -g*Xi gives an LQR gain that is unstable on the true plant."""
    Ac, Bc = X.lqr_matrices(check=False)
    assert np.abs(Ac[3:6, 6:9] - X.P.g * XI).max() < 1e-9, "A_c is not +g*Xi"
    assert np.abs(Ac[3:6, 6:9] + X.P.g * XI).max() > 1.0, "A_c matches the spec's -g*Xi"

    def gain(A, B):
        Ad, Bd = X.discretise(A, B)
        Pi = solve_discrete_are(Ad, Bd, X.Q_HAND, X.R_HAND)
        return np.linalg.solve(X.R_HAND + Bd.T @ Pi @ Bd, Bd.T @ Pi @ Ad)

    Ad_t, Bd_t = X.discretise(Ac, Bc)                    # the TRUE plant
    # C-1 is the sign alone.  (5.6)'s other error, the halved rate block of
    # B_c, cannot be expressed under the 17-state model: B_c has no attitude
    # rows at all (the input reaches attitude through omega, and omega through
    # Omega), which is what test_C2 now pins.
    A_spec = Ac.copy(); A_spec[3:6, 6:9] = -X.P.g * XI
    rho_spec = np.abs(np.linalg.eigvals(Ad_t - Bd_t @ gain(A_spec, Bc))).max()
    rho_corr = np.abs(np.linalg.eigvals(Ad_t - Bd_t @ gain(Ac, Bc))).max()
    assert rho_spec > 1.0, f"spec gain is stable after all: rho={rho_spec}"
    assert rho_corr < 1.0, f"corrected gain is unstable: rho={rho_corr}"


def test_C2_the_model_is_a_cascade_not_a_direct_attitude_input():
    """C-2 under the 17-state model.

    ``delta_dot = om`` is a statement about A_c, and the input reaches nothing
    but the rotor block directly.  The 10-state model's direct
    ``B[6:9,1:4] = diag(om_max)`` encoded "the rate loop is infinitely fast",
    which overstated one-step rate authority against the plant by 24x.
    """
    Ac, Bc = X.lqr_matrices(check=False)
    assert np.abs(Ac[6:9, 9:12] - np.eye(3)).max() < 1e-9, "delta_dot != om"
    assert np.abs(Bc[:12, :]).max() < 1e-9, "B_c still has a direct block"
    assert np.abs(Bc[12:16, :]).max() > 1e-6, "B_c's rotor block is dead"


def test_C2_steady_state_gains_survive_the_cascade():
    """The DC gains the old B_c stood in for must still be exact.

    Both are identities, not fits: (2.11) under the corrected PX4 actuator map
    (A2) is 21.666957, not 25.490538; and the rate loop's steady state is
    om_max exactly -- which is WHY fc drops K_d along with K_i (keeping K_d
    alone leaves a 2 % droop, 9.804 rather than 10).
    """
    import jax.numpy as jnp
    par = X.make_par(1)
    xh, uh = X.plant_to_ctrl(X.hover_state(1, par=par)), X.hover_u(1, par=par)

    def settle(u, n=400):
        x = xh
        for _ in range(n):
            x = X.step_c(x, u)
        return x

    eps = 1e-5
    up, um = uh.at[:, 0].add(eps), uh.at[:, 0].add(-eps)
    daz = float((X.fc(settle(up), up)[0, 5] - X.fc(settle(um), um)[0, 5]) / (2 * eps))
    assert abs(daz - X.DAZ_DC_HOVER) < 1e-4, f"d a_z/d c = {daz}"
    assert abs(daz - 21.666957) < 1e-4
    up, um = uh.at[:, 1].add(eps), uh.at[:, 1].add(-eps)
    dom = float((settle(up)[0, 10] - settle(um)[0, 10]) / (2 * eps))
    assert abs(dom - X.OM_MAX[0]) < 1e-6, f"d om_x/d u_roll = {dom}, want 10"


def test_C3_slung_tension_sign():
    """C-3: the spec's (6.1) gives a hanging load that pushes the airframe up."""
    s = A.SlungScenario(level=10.0)
    ml, e3 = s.m_load, np.array([0.0, 0.0, 1.0])
    for n_hat in (-e3, np.array([0.5, 0.0, -np.sqrt(3) / 2])):
        spec = ml * (X.P.g * (n_hat @ e3))            # as written in (6.1)
        corr = ml * (-X.P.g * (n_hat @ e3))           # as implemented
        truth = ml * X.P.g * abs(n_hat @ e3)
        assert spec < 0 < corr and abs(corr - truth) < 1e-12
    st = X.hover_state(1)
    F = np.asarray(s.wrench_body(jnp.asarray([[0.0, 0.0, -1.0, 0, 0, 0]]), st, 0.0))
    assert F[0, 2] < 0.0


def test_C4_moderated_suites_are_not_disjoint_on_thrust():
    """C-4: thrust, mass, K_w and inertia all overlap between S2 and S3."""
    d = M.disjointness(M.moderate(X.disturbed_spec()),
                       M.moderate(X.ood_spec(), wind=(2.5, 4.0)))
    row = d.set_index("channel")
    assert not bool(row.loc["T", "disjoint"]), "thrust really is disjoint"
    assert bool(row.loc["T", "S2_subset_of_S3"])
    for k in ("m", "Kw", "J"):
        assert not bool(row.loc[k, "disjoint"])
    for k in ("D", "tau", "wind"):
        assert bool(row.loc[k, "disjoint"])


def test_C5_sign_flipped_u_prime_gives_6p86_not_4p65():
    """C-5: the '+' sign is right (kappa_a = 9.0779); the flipped variant gives
    6.8602, not the 4.6464 the spec quotes."""
    th = np.linspace(0.0, 2 * np.pi, 400_001)
    c, s = np.cos(th), np.sin(th)
    f = c ** 8 + s ** 8
    u = s ** 7 * c - c ** 7 * s
    def kappa(up):
        r = f ** (-1 / 8); rp = -f ** (-9 / 8) * u
        rpp = 9 * f ** (-17 / 8) * u ** 2 - f ** (-9 / 8) * up
        a = np.c_[rpp * c - 2 * rp * s - r * c, rpp * s + 2 * rp * c - r * s]
        return float(np.linalg.norm(a, axis=1).max())
    good = 7 * s ** 6 * c ** 2 - s ** 8 + 7 * c ** 6 * s ** 2 - c ** 8
    bad = 7 * s ** 6 * c ** 2 - s ** 8 - 7 * c ** 6 * s ** 2 - c ** 8
    assert abs(kappa(good) - 9.0778770) < 1e-5
    assert abs(kappa(bad) - 6.8602) < 1e-2
    assert abs(kappa(bad) - 4.6464) > 1.0
    assert abs(X.KAPPA_A[5] - 9.0778770) < 1e-6


def test_N1_closed_form_equals_the_arcsin_form():
    """N-1: the two forms of (4.2) agree, but only one is differentiable at 0."""
    rng = np.random.default_rng(1)
    xr = jnp.asarray([[0.0, 0.0, 1.5, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]
                      + [float(X.OM_HOVER)] * 4])
    worst = 0.0
    for _ in range(300):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        th = rng.uniform(1e-6, np.deg2rad(150))
        dlt = 2 * np.sin(th / 2) * ax
        e = jnp.zeros((1, X.NE)).at[0, 6:9].set(jnp.asarray(dlt))
        n = float(np.linalg.norm(dlt))
        ang = 2 * np.arcsin(min(n / 2, 1.0))
        q_arcsin = np.r_[np.cos(ang / 2), dlt / n * np.sin(ang / 2)]
        worst = max(worst, float(jnp.abs(X.e_to_state(e, xr)[0, 6:10]
                                         - jnp.asarray(q_arcsin)).max()))
    assert worst < 1e-12


def test_N2_clamp_gradient_passes_through_on_the_boundary():
    """N-2: derivative 1 on the box, not 0 -- otherwise Q_uu goes singular
    exactly when the line search parks the input on the box."""
    g = jax.grad(lambda x: X._clamp(x, 0.0, 1.0))
    assert float(g(0.0)) == 1.0 and float(g(1.0)) == 1.0
    assert float(g(0.5)) == 1.0
    assert float(g(-0.1)) == 0.0 and float(g(1.1)) == 0.0


def test_N6_naive_subset_containment_would_fail_on_S3():
    """N-6: every S3 multiplicative band lies above 1, so contracting toward 1
    moves it outside the raw band.  A subset test fails on a correct
    moderation."""
    raw = X.ood_spec()
    mod = M.moderate(raw, wind=(2.5, 4.0))
    for k in ("D", "tau"):
        lo, hi = raw[k]; alo, ahi = mod[k]
        assert lo > 1.0, f"{k} raw band straddles 1 after all"
        assert alo < lo, f"{k}: moderated band is inside the raw band"
    assert M.check_moderate(raw, mod, 4.0)      # the correct invariants hold


def test_N7_asym_trim_limit():
    """N-7: the largest trimmable arm-tip mass fraction is 0.0812, so the §6.1
    top level of 0.11 is not flyable with the nominal mixer."""
    tau_avail = X.J_NOM[0, 0] * X.P.K_i[0] * X.P.I_lim
    f_max = tau_avail / (0.174 * X.M_TOT * X.P.g)
    assert abs(tau_avail - 0.28597) < 1e-4
    assert abs(f_max - 0.0812) < 1e-3
    assert A.SCEN_LEVELS["asym"][-1] > f_max
    assert A.moderate_levels(A.SCEN_LEVELS["asym"])[-1] < f_max
