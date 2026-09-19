"""The four preflight checks of §9.4.  **Any failure aborts the run.**

    ros2 run acmpc_controller check_ctbr

1. Frame round trip -- the composed rotation matches the direct construction
   over >= 500 random attitudes, and the tilt is heading-independent.
2. CTBR mapping -- nine assertions on the exit mapping.
3. Thrust calibration -- two nonlinearities compose; T_max is derived from ONE
   module and every consumer is asserted against it.
4. ``actuator_motors`` is publishing -- without it the moment channels are
   unobservable and the dataset is worthless (§6.2).

Checks 1-3 are pure arithmetic and run anywhere.  Check 4 needs a live graph and
degrades to a clearly-labelled SKIP when rclpy is absent, because a check that
silently passes when it cannot run is worse than no check.
"""
from __future__ import annotations

import sys

import numpy as np

from . import frames as F

T_MAX = 34.19432                 # the ONE definition; see check_glue
M_NOM = 2.0643076923076924
G = 9.8066
#: SIM_GZ_EC_MIN / EC_MAX: PX4 maps the normalised actuator command onto
#: [OM_MIN, OM_MAX], so thrust is NOT proportional to c^2 (A2).
OM_MIN, OM_MAX = 150.0, 1000.0
_R = OM_MIN / OM_MAX
U_HOVER = float((np.sqrt(M_NOM * G / T_MAX) - _R) / (1.0 - _R))
T_IDLE = T_MAX * _R ** 2         # 0.769 N at c = 0: 3.8 % of weight
ALLOC_F = 0.0                    # this airframe: c = (1-f) y + f y^2 with f = 0
#: CA_ROTORn_KM: what the PX4 allocator believes, against the Gazebo plugin's
#: momentConstant of 0.016.  A commanded yaw torque realises 32 % of its intent.
K_M_GZ, K_M_PX4 = 0.016, 0.05


def check_frames(n=500, seed=0, tol_q=1e-15, tol_R=5e-15, tol_tilt=1e-17):
    """Check 1.  The quaternion round trip carries the 1e-15 statement; the
    matrix comparison is its shadow and accumulates a few ulp in the triple
    product, so it is held to 5e-15."""
    rng = np.random.default_rng(seed)
    worst_q = worst_R = 0.0
    for _ in range(n):
        q = F.qnormalise(rng.normal(size=4))
        qe = F.px4_quat_to_enu_flu(q)
        back = F.enu_flu_to_px4_quat(qe)
        worst_q = max(worst_q, min(np.abs(back - q).max(), np.abs(back + q).max()))
        R = (F.qrotmat(F.Q_NED_ENU[None])[0] @ F.qrotmat(q[None])[0]
             @ F.qrotmat(F.Q_FRD_FLU[None])[0])
        worst_R = max(worst_R, np.abs(F.qrotmat(qe[None])[0] - R).max())
    q0 = F.qnormalise(np.array([np.cos(0.2), np.sin(0.2), 0.0, 0.0]))
    tilts = [F.tilt_angle(F.qmul(np.array([np.cos(p / 2), 0, 0, np.sin(p / 2)]), q0))
             for p in np.linspace(0, 2 * np.pi, 64)]
    spread = float(np.ptp(tilts))
    ok = worst_q < tol_q and worst_R < tol_R and spread < tol_tilt
    return ok, dict(round_trip=worst_q, composed_vs_direct=worst_R,
                    tilt_heading_spread=spread, n=n)


def check_ctbr_mapping():
    """Check 2.  Nine assertions on the exit mapping."""
    a = []
    om = np.array([0.31, -0.42, 0.17])
    r = F.ctbr_to_px4_rates(om)
    a += [("roll = +om_x", r[0] == om[0]),
          ("pitch = -om_y", r[1] == -om[1]),
          ("yaw = -om_z", r[2] == -om[2])]
    t = F.ctbr_to_px4_thrust(U_HOVER)
    a += [("thrust_body.x = 0", t[0] == 0.0),
          ("thrust_body.y = 0", t[1] == 0.0),
          ("thrust_body.z = -c (down-positive)", abs(t[2] + U_HOVER) < 1e-15)]
    v = np.array([1.0, 2.0, 3.0])
    a += [("NED<->ENU is involutive",
           bool(np.allclose(F.ned_to_enu_vec(F.ned_to_enu_vec(v)), v))),
          ("FRD<->FLU is involutive",
           bool(np.allclose(F.frd_to_flu_vec(F.frd_to_flu_vec(v)), v))),
          ("rate mapping is its own inverse",
           bool(np.allclose(F.ctbr_to_px4_rates(F.ctbr_to_px4_rates(om)), om)))]
    return all(ok for _, ok in a), dict(assertions=a)


def check_thrust_calibration(y_measured=None, tol_pct=10.0):
    """Check 3.  THREE nonlinearities compose on gz_x500: the allocator applies
    ``c = (1-f) y + f y^2`` (f = 0 here), PX4 maps the normalised command onto
    ``Omega = OM_MIN + y (OM_MAX - OM_MIN)``, and the motor gives
    ``F = n K_T Omega^2``.  The middle step is the one that is easy to miss: it
    makes ``y_hover = 0.728742``, not ``sqrt(m g / T_max) = 0.769431`` (A2).

    The calibration factor is ``(y_hover / y_measured)^2``.  Using an
    uncalibrated T_max in the *truth* while the thrust model uses the calibrated
    one biases **every recorded F_ext in the same direction** -- a systematic
    error invisible in any per-sample check.  That is why T_max is derived from
    one module and asserted everywhere else.
    """
    y_hover = U_HOVER
    if y_measured is None:
        return None, dict(y_hover_predicted=y_hover, alloc_f=ALLOC_F,
                          idle_thrust_N=T_IDLE, om_min=OM_MIN,
                          note="no measured hover collective supplied; fly S0 "
                               "and pass --y-measured to calibrate")
    factor = (y_hover / float(y_measured)) ** 2
    err = 100.0 * abs(factor - 1.0)
    return bool(err <= tol_pct), dict(y_hover_predicted=y_hover,
                                      y_measured=float(y_measured),
                                      calibration_factor=factor,
                                      deviation_pct=err,
                                      T_max_calibrated=T_MAX * factor)


def check_actuator_motors(topic="/fmu/out/actuator_motors", timeout_s=3.0,
                          namespace=""):
    """Check 4.  ``actuator_motors`` must be publishing.

    Per §6.2 the PWM block is the only observable channel for a standing moment:
    an integrating rate loop drives a constant external moment out of the rate
    error (measured: 7e-11 rad/s) while the mixer output holds its spread
    (0.0437) indefinitely.  **Refuse to start any moment-producing scenario if
    this fails.**

    The topic is resolved through :mod:`px4_topics`, not subscribed by name.  A
    check that subscribes ``/fmu/out/actuator_motors`` against a PX4 1.16 graph
    reports FAIL on a topic that is publishing perfectly well under
    ``_v1`` -- which sends you looking for a mixer problem that does not exist.
    """
    try:
        import rclpy                                      # noqa: F401
        from rclpy.node import Node
    except Exception as ex:                               # noqa: BLE001
        return None, dict(skipped=True, reason=f"no live ROS graph ({type(ex).__name__})",
                          topic=topic)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    from . import px4_topics as PT
    rclpy.init()
    node = Node("check_actuator_motors")
    try:
        spec = PT.resolve(node, topic, timeout_s=timeout_s, namespace=namespace,
                          required=False)
    except PT.VersionMismatch as ex:
        node.destroy_node(); rclpy.shutdown()
        return False, dict(topic=topic, messages=0, hz=0.0, version_mismatch=str(ex))
    if spec is None:
        node.destroy_node(); rclpy.shutdown()
        return False, dict(topic=topic, messages=0, hz=0.0,
                           reason="no topic matching this base, versioned or not")
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=5)
    got = {"n": 0}
    node.create_subscription(spec.msg_class, spec.topic,
                             lambda _m: got.__setitem__("n", got["n"] + 1), qos)
    t_end = node.get_clock().now().nanoseconds * 1e-9 + timeout_s
    while node.get_clock().now().nanoseconds * 1e-9 < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    hz = got["n"] / timeout_s
    return bool(got["n"] > 0), dict(topic=spec.topic, resolved=str(spec),
                                    messages=got["n"], hz=hz)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    y_meas = None
    if "--y-measured" in argv:
        y_meas = float(argv[argv.index("--y-measured") + 1])
    moment_scenario = "--moment-scenario" in argv

    print("=" * 74)
    print("§9.4 preflight -- any failure aborts the run")
    print("=" * 74)
    results = []
    for name, (ok, info) in (
            ("1. frame round trip", check_frames()),
            ("2. CTBR mapping", check_ctbr_mapping()),
            ("3. thrust calibration", check_thrust_calibration(y_meas)),
            ("4. actuator_motors publishing", check_actuator_motors())):
        tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  [{tag}] {name}")
        if name.startswith("2"):
            for a, v in info["assertions"]:
                print(f"           {'ok ' if v else 'FAIL'}  {a}")
        else:
            for k, v in info.items():
                print(f"           {k}: {v}")
        results.append((name, ok))

    hard = [n for n, ok in results if ok is False]
    skipped = [n for n, ok in results if ok is None]
    print()
    if hard:
        print(f"PREFLIGHT FAILED: {hard}.  Do not record.")
        return 1
    if moment_scenario and any(n.startswith("4") for n in skipped):
        print("REFUSING to start a moment-producing scenario: actuator_motors "
              "could not be verified.  Without it the moment channels are "
              "unobservable and the dataset is worthless (§6.2).")
        return 2
    if skipped:
        print(f"PREFLIGHT PASSED with skips: {skipped}")
        print("  A skip is not a pass.  Re-run against a live graph before "
              "recording, and pass --moment-scenario to make check 4 blocking.")
        return 0
    print("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
