"""check_px4 -- the PX4 message-versioning preflight.  **Run it before every demo.**

    ros2 run acmpc_controller check_px4
    ros2 run acmpc_controller check_px4 --namespace /px4_1 --timeout 20

PX4 1.16 versions its uORB messages, and the bridge advertises the version on
the wire (``/fmu/out/vehicle_odometry_v1``).  Every failure mode of a mismatch
is silent:

  * subscribe to the unversioned name against a versioned graph -> the callback
    is never invoked.  No error.  The controller sits on ``self.state is None``
    forever and logs nothing, because it is *waiting*, which looks the same as
    starting up.
  * publish the unversioned name -> PX4 never sees a setpoint, offboard is
    rejected after 0.5 s, and the vehicle either never arms or falls out of
    offboard mid-flight.
  * match the name but not the ABI (``px4_msgs`` from a different tag) -> the
    message deserialises into plausible-looking garbage.  A quaternion that is
    not a quaternion is the worst of the three because the run *looks* alive.

So this check does not test a guess.  It reads the graph, prints what is there,
resolves each topic the workspace uses the same way the nodes do, and reports
the version it landed on.  If the census and the resolution disagree with your
expectations, the census is right.
"""
from __future__ import annotations

import sys

from . import px4_topics as PT

#: The topics the demo needs, and whether it can proceed without them.
REQUIRED = (("/fmu/out/vehicle_odometry", True,
             "state for the controller, the estimator and the plot"),
            ("/fmu/in/vehicle_rates_setpoint", True, "CTBR output"),
            ("/fmu/in/offboard_control_mode", True, "offboard heartbeat"),
            ("/fmu/in/vehicle_command", True, "arm / set-mode"),
            ("/fmu/out/actuator_motors", False,
             "PWM block -- the ONLY observable channel for a standing moment "
             "(§6.2).  Without it the RDP's moment outputs are unobservable "
             "and the payload demo measures nothing about torque."),
            ("/fmu/out/vehicle_status", False, "arming and nav state, for logging"))


def probe(node, namespace="", timeout_s=15.0):
    """-> (ok, rows).  One row per required topic: (base, spec_or_error, fatal)."""
    rows, ok = [], True
    first = True
    for base, fatal, why in REQUIRED:
        try:
            spec = PT.resolve(node, base, timeout_s if first else 2.0, namespace,
                              required=False)
        except PT.VersionMismatch as ex:
            rows.append((base, f"VERSION MISMATCH\n{ex}", fatal, why))
            ok = ok and not fatal
            first = False
            continue
        first = False
        if spec is None:
            rows.append((base, None, fatal, why))
            ok = ok and not fatal
        else:
            rows.append((base, spec, fatal, why))
    return ok, rows


def main(argv=None):                                       # pragma: no cover
    argv = sys.argv[1:] if argv is None else argv
    ns = argv[argv.index("--namespace") + 1] if "--namespace" in argv else ""
    tmo = float(argv[argv.index("--timeout") + 1]) if "--timeout" in argv else 15.0

    try:
        import rclpy
        from rclpy.node import Node
    except Exception as ex:                               # noqa: BLE001
        print(f"check_px4: no ROS 2 environment ({type(ex).__name__}).  This "
              f"check needs a live graph; there is nothing useful it can do "
              f"without one.", file=sys.stderr)
        return 2

    print("=" * 78)
    print("check_px4 -- PX4 topic and message-version preflight")
    print("=" * 78)
    try:
        import px4_msgs.msg as PM
        vs = sorted(n for n in dir(PM) if n.startswith("VehicleOdometry"))
        print(f"\npx4_msgs importable; VehicleOdometry* classes present: {vs}")
    except Exception as ex:                               # noqa: BLE001
        print(f"\npx4_msgs is NOT importable ({type(ex).__name__}: {ex}).")
        print("  source your PX4 workspace:  source ~/px4_ws/install/setup.bash")
        return 1

    rclpy.init()
    node = Node("check_px4")
    print(f"\nwaiting up to {tmo:g} s for the graph "
          f"(namespace {ns or '<none>'}) ...\n")
    ok, rows = probe(node, ns, tmo)

    print("PX4 topic census:")
    print(PT.graph_report(node.get_topic_names_and_types(), ns))

    print("\nresolution:")
    versions = set()
    for base, spec, fatal, why in rows:
        if isinstance(spec, PT.TopicSpec):
            print(f"  [PASS] {base}\n         -> {spec}")
            versions.add(spec.version)
        elif spec is None:
            print(f"  [{'FAIL' if fatal else 'WARN'}] {base}  -- not advertised")
            print(f"         {why}")
        else:
            print(f"  [FAIL] {base}")
            for ln in str(spec).splitlines():
                print(f"         {ln}")

    if len(versions) > 1:
        print(f"\n  note: this graph carries more than one message generation "
              f"{sorted(versions)}.\n  That is legal during a PX4 transition and "
              f"the nodes resolve each topic on its own,\n  but it is worth "
              f"knowing about before you read the results.")

    node.destroy_node()
    rclpy.shutdown()
    print()
    if ok:
        print("check_px4 PASSED -- every required topic resolved.")
        return 0
    print("check_px4 FAILED.  Fix the above before flying; every symptom of a\n"
          "version mismatch downstream of here is silent.\n"
          "  * no topics at all      -> the uXRCE-DDS agent is not running:\n"
          "                             MicroXRCEAgent udp4 -p 8888\n"
          "  * topics but no types   -> px4_msgs is from a different PX4 tag;\n"
          "                             rebuild it from the firmware's tag\n"
          "  * only /fmu/out present -> PX4 is up but nothing subscribes the\n"
          "                             /fmu/in side yet; start the agent first")
    return 1


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
