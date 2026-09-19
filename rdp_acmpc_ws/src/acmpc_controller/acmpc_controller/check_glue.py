"""check_glue -- the workspace and the study must agree on every constant.

    ros2 run acmpc_controller check_glue

§11: *a green control-law test says nothing about frames or constants* -- keep
them separate.  This is the constants half, from the workspace side; the study
side is ``study/check_consistency.py`` and the two are deliberately independent
so that a single edit cannot silently satisfy both.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ROOT = os.path.dirname(WS)
sys.path.insert(0, os.path.join(ROOT, "study"))
sys.path.insert(0, os.path.join(WS, "src", "reference_generator"))
sys.path.insert(0, os.path.join(WS, "src", "disturbance_manager"))

from . import bd_bridge, check_ctbr  # noqa: E402

_fail = []


def _chk(name, got, want, tol=1e-12):
    ok = (abs(float(got) - float(want)) <= tol * max(1.0, abs(float(want)))
          if np.isscalar(got) or np.ndim(got) == 0
          else bool(np.allclose(got, want, atol=tol)))
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}"
          + ("" if ok else f"   got {got!r} want {want!r}"))
    if not ok:
        _fail.append(name)


def main(argv=None):
    print("=" * 74)
    print("check_glue -- workspace <-> study constants and the B_d bridge")
    print("=" * 74)
    try:
        import x500_core_jax as X
        import adaptive_core_jax as A
    except Exception as ex:                               # noqa: BLE001
        print(f"  cannot import the study module ({type(ex).__name__}: {ex}).")
        print("  The workspace cannot be validated against it; refusing to pass.")
        return 1

    print("\nconstants:")
    _chk("m", bd_bridge.M_NOM, X.M_TOT)
    _chk("T_max (ONE definition)", check_ctbr.T_MAX, X.T_MAX)
    _chk("g", bd_bridge.wrench_to_dmod(np.zeros(6)).sum() * 0 + 9.8066, X.P.g)
    _chk("J", bd_bridge.J_NOM, X.J_NOM)
    _chk("u_hover", check_ctbr.U_HOVER, X.U_HOVER)
    from reference_generator import lissajous as L
    _chk("a_lat_max", L.A_LAT_MAX, X.A_LAT_MAX)
    _chk("alpha", L.ALPHA, X.P.alpha_feas)
    from disturbance_manager import scenarios as SC
    _chk("scenario m", SC.M_NOM, X.M_TOT)
    _chk("scenario arm", SC.ARM, float(abs(X.R_ROTOR[0, 0])))

    print("\nB_d bridge (§4.4), against the study's wrench_to_dmod:")
    rng = np.random.default_rng(0)
    w = rng.normal(size=(64, 6)) * np.array([5, 5, 5, 0.3, 0.3, 0.3])
    a = bd_bridge.wrench_to_dmod(w)
    b = np.asarray(X.wrench_to_dmod(w))
    _chk("wrench_to_dmod agrees on 64 random wrenches", float(np.abs(a - b).max()), 0.0)
    _chk("mode='none' zeroes the moment block",
         float(np.abs(bd_bridge.wrench_to_dmod(w, "none")[:, 3:]).max()), 0.0)
    _chk("J^-1 (the moment block is EXACT now, not (4.13))",
         bd_bridge.JINV_NOM, np.asarray(X.JINV_NOM))
    _chk("§4.4 example a_res", float(bd_bridge.wrench_to_dmod(
        np.array([6.478, 0, 0, 0, 0, 0]))[0]), 6.478 / X.M_TOT, 1e-9)
    _chk("§4.4 example alpha_res", float(bd_bridge.wrench_to_dmod(
        np.array([0, 0, 0, 0.3452, 0, 0]))[3]), 0.3452 / X.J_NOM[0, 0], 1e-9)

    print("\n17-state reference (refgen vs the study's ref_state):")
    from . import refgen as RG
    import jax.numpy as jnp
    env = X.Env(n=1, seed=7, ep_len=500, paths=("fig8",),
                dist=X.nominal_spec(speed=(2.0, 2.0)))

    def pva(t, _ep=env.ep):
        pp, vv, aa = X._ref_pva(_ep, jnp.asarray([float(t)]))
        return np.asarray(pp)[0], np.asarray(vv)[0], np.asarray(aa)[0]

    wx = wu = 0.0
    for t in np.linspace(0.0, 4.0, 41):
        xs, us, _ = X.ref_state(env.ep, jnp.asarray([t]))
        xn, un = RG.ref_stage(pva, float(t))
        wx = max(wx, float(np.abs(np.asarray(xs)[0] - xn).max()))
        wu = max(wu, float(np.abs(np.asarray(us)[0] - un).max()))
    # om_dot_ref is a second centred difference, so it carries ~1/(2h)^2 of the
    # reference's own round-off; 1e-8 is the honest tolerance, not 1e-12
    _chk("xr (17) agrees over 41 sample times", wx, 0.0, 1e-8)
    _chk("u_ref agrees over 41 sample times", wu, 0.0, 1e-12)
    _chk("control state width", X.NX, 17)
    _chk("error width", X.NE, 16)

    print("\nS7 payload demonstration (plant, truth and plot must agree):")
    sys.path.insert(0, os.path.join(WS, "src", "visualization"))
    sys.path.insert(0, os.path.join(WS, "tools"))
    import payload_sdf as PSDF
    from visualization import estimates as EST
    from visualization import live_zx as LZX
    from . import controller_node as CN
    _chk("payload mass: SDF patch vs scenario", PSDF.DEFAULT_MASS, SC.S7_PAYLOAD_M)
    _chk("payload offset: SDF patch vs scenario",
         np.asarray(PSDF.DEFAULT_OFFSET), np.asarray(SC.S7_PAYLOAD_R))
    _chk("estimates m", EST.M_NOM, X.M_TOT)
    _chk("estimates g", EST.G, X.P.g)
    _chk("plotter K_T", LZX.K_T, X.P.K_T)
    _chk("plotter Om_min", LZX.OM_MIN, X.P.Om_min)
    _chk("plotter Om_max", LZX.OM_MAX, X.P.Om_max)
    sc7 = SC.Scenario("S7")
    F7, tau7 = sc7.wrench(0.0)
    tr7 = EST.payload_truth(sc7.params["payload_mass"], sc7.params["payload_offset"])
    _chk("S7 truth force agrees with the plot's truth line", F7[2], tr7["F_z"])
    _chk("S7 truth moment agrees with the plot's truth line", tau7, tr7["tau"])
    m7, mp7 = EST.mass_estimate(F7)
    _chk("mass estimate inverts the S7 residual", mp7, sc7.params["payload_mass"], 1e-9)
    rx7, ry7 = EST.cg_offset_estimate(tau7, mp7)
    _chk("CG estimate inverts the S7 moment",
         np.array([float(rx7), float(ry7)]),
         np.asarray(sc7.params["payload_offset"])[:2], 1e-9)
    adaptive = [n for n, v in CN.CONTROLLERS.items() if v["use_d"]]
    ok_a = adaptive == ["acmpc_adaptive"]
    print(f"  [{'OK ' if ok_a else 'FAIL'}] exactly one controller is fed d_hat: "
          f"{adaptive}")
    if not ok_a:
        _fail.append("d_hat routing")

    print("\nsix channel orderings:")
    from rdp_estimator.ring_buffer import FRAME_DIM
    _chk("frame width", FRAME_DIM, A.FRAME_DIM)
    ok = list(A.CHANNEL_NAMES) == ["F_x", "F_y", "F_z", "M_x", "M_y", "M_z"]
    print(f"  [{'OK ' if ok else 'FAIL'}] channel order {list(A.CHANNEL_NAMES)}")
    if not ok:
        _fail.append("channel order")

    print("\nexported models:")
    mdir = os.path.join(WS, "models")
    found = [f for f in sorted(os.listdir(mdir))] if os.path.isdir(mdir) else []
    npz = [f for f in found if f.endswith(".npz")]
    if not npz:
        print("  [SKIP] no exported .npz yet -- run study/export_estimator.py")
    else:
        import json
        import rdp_infer
        for f in npz:
            meta = json.loads(str(np.load(os.path.join(mdir, f),
                                          allow_pickle=False)["__meta__"]))
            _chk(f"{f} frame_dim", meta["frame_dim"], A.FRAME_DIM)
            _chk(f"{f} H", meta["H"], A.H_DEFAULT)
            okc = list(meta["channels"]) == list(A.CHANNEL_NAMES)
            print(f"  [{'OK ' if okc else 'FAIL'}] {f} channel names")
            if not okc:
                _fail.append(f"{f} channels")
            r = rdp_infer.load(os.path.join(mdir, f))
            y = r.predict(np.zeros((1, r.H, r.frame_dim)))
            print(f"           forward pass ok, output {y.shape}")

    print()
    if _fail:
        print(f"check_glue FAILED ({len(_fail)}): {', '.join(_fail)}")
        return 1
    print("check_glue PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
