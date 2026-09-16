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
             # J now carries the rotors' OWN inertia from the SDF, not just
             # their point masses (A6): +0.006 % on xx, +0.44 % on yy, +0.24 % on zz
             ("J_xx", X.J_NOM[0, 0], 0.023832494), ("J_yy", X.J_NOM[1, 1], 0.023935416),
             ("J_zz", X.J_NOM[2, 2], 0.043999950), ("f_max", X.F_MAX, 8.54858),
             ("T_max", X.T_MAX, 34.19432), ("T/W", X.TW, 1.689122),
             # u_hover under the PX4 actuator map, Omega = 150 + 850c (A2)
             ("u_hover", X.U_HOVER, 0.728742),
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
    # A3: the PX4 allocator inverts with CA_ROTORn_KM = 0.05, not the Gazebo
    # plugin's 0.016, so its condition number is 20 and a commanded yaw torque
    # realises 0.016/0.05 = 32 % of its intent
    assert abs(float(np.linalg.cond(X.M_CTRL)) - 20.0) < 1e-6
    f = np.linalg.inv(X.M_CTRL) @ np.array([0.0, 0.0, 0.0, 0.1])
    assert abs((X.M_NOM @ f)[3] / 0.1 - X.P.k_m / X.P.k_m_ctrl) < 1e-9


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
    # A2: with the PX4 idle floor the hover slope is 21.666957, not 25.490538.
    # The pure-square form overstates control effectiveness by 17.65 %, which
    # propagates straight into K_LQR and the terminal matrix.
    assert abs(fd - 21.666957) < 1e-4
    assert abs(X.DAZ_DC_HOVER - 21.666957) < 1e-4
    assert abs(X.T_MAX / X.M_TOT - 16.5645) < 1e-3     # the linear value, for contrast
    # the analytic form: 2 n K_T Om_hover (Om_max - Om_min) / m
    want = (2 * X.P.n_rotor * X.P.K_T * X.OM_HOVER
            * (X.P.Om_max - X.P.Om_min) / X.M_TOT)
    assert abs(fd - want) < 1e-6
