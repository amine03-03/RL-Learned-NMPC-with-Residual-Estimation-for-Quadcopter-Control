"""T-1, T-2, T-5 -- constants, allocation, hover linearisation.

Every one of these exists because the corresponding error is silent.
"""
import numpy as np
import jax.numpy as jnp
import pytest

import x500_core_jax as X

REL = 1e-6


def test_T1_constants():
    """T-1: m, cg0, J, T_max, T/W, u_hover, a_lat_max to 1e-6 relative."""
    cases = [("m", X.M_TOT, 2.064308), ("cg0_z", X.CG_NOM[2], 0.001869131),
             ("J_xx", X.J_NOM[0, 0], 0.023830955), ("J_yy", X.J_NOM[1, 1], 0.023830955),
             ("J_zz", X.J_NOM[2, 2], 0.043893959), ("f_max", X.F_MAX, 8.54858),
             ("T_max", X.T_MAX, 34.19432), ("T/W", X.TW, 1.689122),
             ("u_hover", X.U_HOVER, 0.769431),
             ("a_lat_max", X.A_LAT_MAX, 13.349711)]
    for name, got, want in cases:
        assert abs(got - want) <= REL * max(1.0, abs(want)), f"{name}: {got} != {want}"
    assert abs(X.CG_NOM[0]) < 1e-15 and abs(X.CG_NOM[1]) < 1e-15
    assert abs(X.P.alpha_feas * X.A_LAT_MAX - 8.0098) < 1e-3


def test_T2_allocation():
    """T-2: J off-diagonals, M and M^-1, cond(M) = 62.5."""
    assert np.abs(X.J_NOM - np.diag(np.diag(X.J_NOM))).max() < 1e-15
    M_spec = np.array([[1, 1, 1, 1], [-.174, .174, .174, -.174],
                       [-.174, .174, -.174, .174], [-.016, -.016, .016, .016]])
    assert np.abs(X.M_NOM - M_spec).max() < 1e-12
    Minv_spec = np.array(
        [[.25, -1.436781609, -1.436781609, -15.625],
         [.25, 1.436781609, 1.436781609, -15.625],
         [.25, 1.436781609, -1.436781609, 15.625],
         [.25, -1.436781609, 1.436781609, 15.625]])
    assert np.abs(X.MINV_NOM - Minv_spec).max() < 1e-8
    assert abs(float(np.linalg.cond(X.M_NOM)) - 62.5) < 1e-6
    # the condition number is entirely the yaw row: k_m / arm
    assert abs(X.P.k_m / 0.174 - 0.0920) < 1e-3


def test_T5_hover_linearisation_of_collective():
    """T-5: d a_z / d c at hover = 25.490538, by finite difference of `fc`.

    Not T_max/m = 16.5648: the motor model is velocity-type, so thrust is
    quadratic in the command and the hover slope carries a factor 2*u_hover.
    """
    par = X.make_par(1)
    x = X.plant_to_ctrl(X.hover_state(1, par=par))
    u = X.hover_u(1, par=par)
    h = 1e-6
    fd = float((X.fc(x, u.at[:, 0].add(h))[0, 5]
                - X.fc(x, u.at[:, 0].add(-h))[0, 5]) / (2 * h))
    assert abs(fd - 25.490538) < 1e-4
    assert abs(X.DAZ_DC_HOVER - 25.490538) < 1e-4
    assert abs(X.T_MAX / X.M_TOT - 16.5645) < 1e-3     # the wrong value, for contrast
    assert abs(fd / (X.T_MAX / X.M_TOT) - 2 * X.U_HOVER) < 1e-5
