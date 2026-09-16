"""check_consistency.py -- the study and the ROS workspace must agree.

§11: *a green control-law test says nothing about frames or constants* -- keep
them separate.  This script asserts that the study module, the exported RDP
metadata and the workspace configuration agree on m, the calibrated T_max, J,
the arm length, K_T, k_m, and the six channel orderings.

Run:  python check_consistency.py            (exit 0 = agreement)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adaptive_core_jax as A       # noqa: E402
import x500_core_jax as X           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "rdp_acmpc_ws")

#: **Derive T_max from one module only.**  Everything else reads it from here.
#: Using an uncalibrated T_max in the wrench truth while the thrust model uses
#: the calibrated one biases *every recorded F_ext in the same direction* -- a
#: systematic error invisible in any per-sample check (§9.4 check 3).
STUDY_CONSTANTS = dict(
    m=float(X.M_TOT), T_max=float(X.T_MAX), K_T=float(X.P.K_T), k_m=float(X.P.k_m),
    k_m_ctrl=float(X.P.k_m_ctrl), om_min=float(X.P.Om_min),
    arm=0.174, J_xx=float(X.J_NOM[0, 0]), J_yy=float(X.J_NOM[1, 1]),
    J_zz=float(X.J_NOM[2, 2]), g=float(X.P.g), u_hover=float(X.U_HOVER),
    dt_c=float(X.P.dt_c), om_max=[float(v) for v in X.OM_MAX],
    channels=list(A.CHANNEL_NAMES), units=list(A.CHANNEL_UNITS),
    frame_dim=A.FRAME_DIM, H=A.H_DEFAULT,
)

TOL = 1e-9
_fail = []


def _chk(name, got, want, tol=TOL):
    if isinstance(want, (list, tuple)):
        ok = list(got) == list(want)
        d = "" if ok else f" got {got} want {want}"
    else:
        ok = abs(float(got) - float(want)) <= tol * max(1.0, abs(float(want)))
        d = "" if ok else f" got {got!r} want {want!r}"
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{d}")
    if not ok:
        _fail.append(name)
    return ok


def check_config():
    """The workspace YAML must not re-declare a constant with a different value."""
    path = os.path.join(WS, "config", "acmpc.yaml")
    if not os.path.exists(path):
        print(f"  [SKIP] {os.path.relpath(path, ROOT)} not present")
        return
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    v = cfg.get("/**", {}).get("ros__parameters", cfg)
    veh = v.get("vehicle", {})
    for k in ("m", "T_max", "K_T", "k_m", "arm", "g"):
        if k in veh:
            _chk(f"config acmpc.yaml vehicle.{k}", veh[k], STUDY_CONSTANTS[k], 1e-6)
    if "u_hover" in veh:
        _chk("config acmpc.yaml vehicle.u_hover", veh["u_hover"],
             STUDY_CONSTANTS["u_hover"], 1e-6)
    if "J" in veh:
        for i, ax in enumerate(("xx", "yy", "zz")):
            _chk(f"config acmpc.yaml vehicle.J[{ax}]", veh["J"][i],
                 STUDY_CONSTANTS[f"J_{ax}"], 1e-6)
    if "control_rate_hz" in v:
        _chk("config acmpc.yaml control_rate_hz", 1.0 / v["control_rate_hz"],
             STUDY_CONSTANTS["dt_c"], 1e-9)


def check_exported_models():
    """Every exported .npz must carry the same channel order and window."""
    mdir = os.path.join(WS, "models")
    if not os.path.isdir(mdir):
        print(f"  [SKIP] {os.path.relpath(mdir, ROOT)} not present")
        return
    found = [f for f in sorted(os.listdir(mdir)) if f.endswith(".npz")]
    if not found:
        print("  [SKIP] no exported .npz models yet (run export_estimator.py)")
        return
    for f in found:
        meta = json.loads(str(np.load(os.path.join(mdir, f),
                                      allow_pickle=False)["__meta__"]))
        _chk(f"{f} channels", meta["channels"], STUDY_CONSTANTS["channels"])
        _chk(f"{f} units", meta["units"], STUDY_CONSTANTS["units"])
        _chk(f"{f} frame_dim", meta["frame_dim"], STUDY_CONSTANTS["frame_dim"])
        _chk(f"{f} H", meta["H"], STUDY_CONSTANTS["H"])
        lat = meta.get("latency_ms_numpy")
        if lat:
            adm = lat["p95"] < 20.0
            print(f"  [{'OK ' if adm else 'FAIL'}] {f} p95 latency "
                  f"{lat['p95']:.3f} ms against the 20 ms period")
            if not adm:
                _fail.append(f"{f} latency")


def check_internal():
    """The study's own derived constants, against §2 and against each other."""
    _chk("m = m_b + 4 m_r", X.M_TOT, 2.064308, 1e-6)
    _chk("T_max = 4 K_T Om_max^2", X.T_MAX, 4 * X.P.K_T * X.P.Om_max ** 2)
    # A2: the PX4 actuator map, Omega = Om_min + c (Om_max - Om_min)
    _r = X.P.Om_min / X.P.Om_max
    _chk("u_hover (PX4 actuator map)", X.U_HOVER,
         (np.sqrt(X.M_TOT * X.P.g / X.T_MAX) - _r) / (1.0 - _r))
    _chk("u_hover is NOT sqrt(mg/T_max)",
         abs(X.U_HOVER - np.sqrt(X.M_TOT * X.P.g / X.T_MAX)) > 0.04, True)
    _chk("idle thrust = 3.8 % of weight", X.T_IDLE / (X.M_TOT * X.P.g), 0.038, 1e-2)
    _chk("a_lat_max", X.A_LAT_MAX,
         np.sqrt((X.T_MAX / X.M_TOT) ** 2 - X.P.g ** 2))
    _chk("d a_z/d c = 2 n K_T Om_h (Om_max-Om_min)/m", X.DAZ_DC_HOVER,
         2 * X.P.n_rotor * X.P.K_T * X.OM_HOVER
         * (X.P.Om_max - X.P.Om_min) / X.M_TOT)
    _chk("cond(M_ctrl) = 20 (PX4 CA_ROTORn_KM)",
         float(np.linalg.cond(X.M_CTRL)), 20.0, 1e-9)
    _chk("yaw authority realised", X.P.k_m / X.P.k_m_ctrl, 0.32, 1e-9)
    _chk("cond(M) = 62.5", float(np.linalg.cond(X.M_NOM)), 62.5, 1e-9)
    _chk("J off-diagonals", float(np.abs(X.J_NOM - np.diag(np.diag(X.J_NOM))).max()),
         0.0, 1e-15)
    _chk("arm length", float(abs(X.R_ROTOR[0, 0])), 0.174)
    _chk("wrench->dmod force = 1/m", float(X.wrench_to_dmod(
        np.asarray([[1.0, 0, 0, 0, 0, 0]]))[0, 0]), 1.0 / X.M_TOT)
    # (4.9) and (4.10) must agree on the nominal plant: both identically zero
    par = X.make_par(2)
    s, u = X.hover_state(2, par=par), X.hover_u(2, par=par)
    _chk("T-8 residual zero", float(np.abs(X.true_disturbance(s, u, par)).max()),
         0.0, 1e-9)
    _chk("T-8 wrench zero",
         float(np.abs(X.external_wrench(s, u, par, X.fp(s, u, par))).max()), 0.0, 1e-9)
    # C-1/C-2 cannot regress: lqr_matrices asserts internally
    X.lqr_matrices(check=True)
    print("  [OK ] hover linearisation matches the corrected (5.6) (C-1, C-2)")
    _chk("LQR is stabilising",
         float(np.abs(np.linalg.eigvals(X.AD_HOVER - X.BD_HOVER @ X.K_LQR)).max()) < 1.0,
         True)


def main():
    X.banner("check_consistency.py -- study <-> workspace agreement")
    print("\nstudy internals:")
    check_internal()
    print("\nworkspace configuration:")
    check_config()
    print("\nexported models:")
    check_exported_models()
    print()
    if _fail:
        print(f"FAILED ({len(_fail)}): {', '.join(_fail)}")
        return 1
    print("all consistency checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
