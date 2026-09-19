"""ACMPC controller node (§9.11).

One 50 Hz tick: read state, build the 17-state reference, convert the estimate
through B_d, solve, publish the PX4 setpoint.  Frame conversion happens at this
boundary and nowhere else: everything inside the package is ENU/FLU.

Three controllers share this node and are selected by the ``controller``
parameter -- they are the comparison of §9.8 and of the payload demonstration:

    nmpc1            mode=nmpc1, use_d=False   the hand-tuned incumbent
    acmpc            mode=acmpc, use_d=False   learned cost map, nominal model
    acmpc_adaptive   mode=acmpc, use_d=True    learned cost map + RDP residual

Only the third sees ``/rdp/disturbance_estimate``.  That is the whole point of
the comparison, so the routing is a parameter of the controller and not a wiring
difference between launch files, where it could silently disagree with the label
on the plot.

Every PX4 topic is resolved at run time through :mod:`px4_topics`.  PX4 1.16
versions its uORB messages and a hard-coded ``/fmu/out/vehicle_odometry`` gets a
subscription that is never called -- no error, no warning, a vehicle that never
arms and a plot that stays empty.
"""
from __future__ import annotations

import sys

import numpy as np

from . import frames as F
from . import px4_topics as PT
from . import refgen as RG
from .controller import ACMPCController

#: (mode, use_d) for each named controller.  One table, used by the node, the
#: launch file and the demo driver, so the label on the plot cannot disagree
#: with the wiring behind it.
CONTROLLERS = {
    "nmpc1": dict(mode="nmpc1", use_d=False),
    "acmpc": dict(mode="acmpc", use_d=False),
    "acmpc_adaptive": dict(mode="acmpc", use_d=True),
    "pid": dict(mode="pid", use_d=False),
}

#: PX4 MAVLink command ids used for the offboard handshake.
VEHICLE_CMD_DO_SET_MODE = 176
VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6.0


def main(args=None):                                       # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from geometry_msgs.msg import WrenchStamped
        from std_msgs.msg import Float64MultiArray, String
    except Exception as ex:                               # noqa: BLE001
        print(f"acmpc_controller: no ROS 2 / px4_msgs environment "
              f"({type(ex).__name__}).  The control core is importable and "
              f"testable without one; see ACMPCController.", file=sys.stderr)
        return 1
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                           ReliabilityPolicy)
    from geometry_msgs.msg import WrenchStamped
    from std_msgs.msg import Float64MultiArray, String

    sys.path.insert(0, "src/reference_generator")
    from reference_generator import lissajous as L

    class ControllerNode(Node):
        def __init__(self):
            super().__init__("acmpc_controller")
            for k, v in (("controller", "acmpc"), ("mode", ""), ("use_d", True),
                         ("horizon", 10), ("n_iter", 10), ("rate_hz", 50.0),
                         ("reference", "hold"),
                         ("p_hold", [0.0, 0.0, 1.5]),
                         ("A", 1.2), ("B", 0.9), ("omega", 1.0), ("z0", 1.5),
                         ("auto_arm", True), ("arm_after_s", 1.5),
                         ("px4_namespace", ""), ("topic_timeout_s", 30.0)):
                self.declare_parameter(k, v)
            g = lambda k: self.get_parameter(k).value

            name = g("controller")
            if name not in CONTROLLERS:
                raise SystemExit(
                    f"unknown controller {name!r}; expected one of "
                    f"{sorted(CONTROLLERS)}")
            spec = dict(CONTROLLERS[name])
            if g("mode"):                      # explicit override, for sweeps
                spec["mode"] = g("mode")
            spec["use_d"] = bool(g("use_d")) and spec["use_d"]
            self.cname = name

            # -- reference, checked BEFORE anything is flown ---------------- #
            self.ref_mode = g("reference")
            if self.ref_mode not in ("hold", "lissajous"):
                raise SystemExit(f"reference must be 'hold' or 'lissajous', "
                                 f"got {self.ref_mode!r}")
            self.p_hold = PT.as_floats(g("p_hold"), 3, "p_hold")
            ok, pv, pa, budget = (
                L.check_feasible(mode=L.HOLD) if self.ref_mode == "hold"
                else L.check_feasible(g("A"), g("B"), g("omega")))
            if not ok:
                raise SystemExit(
                    f"reference is INFEASIBLE: peak demand {pa:.3f} m/s^2 "
                    f"exceeds alpha*a_lat_max = {budget:.4f}.  Refusing to fly "
                    f"it; a benchmark built on an infeasible reference measures "
                    f"the reference, not the controller.")
            self.get_logger().info(
                f"controller {name!r} -> mode={spec['mode']} use_d={spec['use_d']}; "
                f"reference {self.ref_mode} "
                + (f"at {self.p_hold.tolist()}" if self.ref_mode == "hold"
                   else f"feasible: peak_a {pa:.3f} <= {budget:.4f} m/s^2"))
            self.ctrl = ACMPCController(horizon=int(g("horizon")),
                                        n_iter=int(g("n_iter")), **spec)

            # -- PX4 topics, resolved from the live graph ------------------- #
            ns, tmo = g("px4_namespace"), float(g("topic_timeout_s"))
            odom = PT.resolve(self, "/fmu/out/vehicle_odometry", tmo, ns)
            rates = PT.resolve(self, "/fmu/in/vehicle_rates_setpoint", 5.0, ns,
                               required=False)
            ocm = PT.resolve(self, "/fmu/in/offboard_control_mode", 5.0, ns,
                             required=False)
            vcmd = PT.resolve(self, "/fmu/in/vehicle_command", 5.0, ns,
                              required=False)
            # An input topic with no subscriber yet is not an error -- the
            # bridge may create it lazily -- but we still need a message CLASS,
            # so fall back to the unversioned import and say so loudly.
            rates = rates or self._fallback("VehicleRatesSetpoint",
                                            f"{ns}/fmu/in/vehicle_rates_setpoint")
            ocm = ocm or self._fallback("OffboardControlMode",
                                        f"{ns}/fmu/in/offboard_control_mode")
            vcmd = vcmd or self._fallback("VehicleCommand",
                                          f"{ns}/fmu/in/vehicle_command")

            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
            self.state = None
            self.q_last = np.array([1.0, 0.0, 0.0, 0.0])
            self.d_hat = np.zeros(6)
            self.create_subscription(odom.msg_class, odom.topic, self.on_odom, qos)
            self.get_logger().info(f"state from {odom}")
            # Keep the resolved CLASSES.  Re-importing px4_msgs.msg.X later in
            # the tick would quietly undo the version resolution done here and
            # publish the wrong ABI onto a versioned topic.
            self.RatesSetpoint = rates.msg_class
            self.OffboardMode = ocm.msg_class
            self.VehicleCommand = vcmd.msg_class
            self.pub_rates = self.create_publisher(rates.msg_class, rates.topic, 10)
            self.pub_mode = self.create_publisher(ocm.msg_class, ocm.topic, 10)
            self.pub_cmd = self.create_publisher(vcmd.msg_class, vcmd.topic, 10)
            self.get_logger().info(f"setpoints to {rates}; mode to {ocm}; "
                                   f"commands to {vcmd}")

            if spec["use_d"]:
                self.create_subscription(WrenchStamped,
                                         "/rdp/disturbance_estimate", self.on_d, 10)
            else:
                self.get_logger().info(
                    f"{name}: NOT subscribing to /rdp/disturbance_estimate -- "
                    f"this controller flies the nominal model by design")

            # what the estimator needs from the control side (u, u_ref, p_ref)
            self.pub_aux = self.create_publisher(Float64MultiArray,
                                                 "/controller/frame_aux", 10)
            latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
            self.pub_name = self.create_publisher(String, "/experiment/controller",
                                                  latched)
            m = String(); m.data = name
            self.pub_name.publish(m)

            self.t0 = None
            self.armed = False
            self.n_tick = 0
            self.auto_arm = bool(g("auto_arm"))
            self.arm_after = float(g("arm_after_s"))
            self.create_timer(1.0 / float(g("rate_hz")), self.tick)

        def _fallback(self, cls_name, topic):
            """Import ``px4_msgs.msg.<cls_name>`` when the graph has no opinion yet."""
            cls = PT.import_msg(f"px4_msgs/msg/{cls_name}")
            if cls is None:
                raise SystemExit(
                    f"px4_msgs has no {cls_name}, and nothing on the graph "
                    f"advertises {topic}.  The installed px4_msgs does not match "
                    f"the firmware; see px4_topics.VersionMismatch.")
            self.get_logger().warn(
                f"{topic} not advertised yet -- publishing unversioned "
                f"px4_msgs/msg/{cls_name}.  If PX4 is 1.16+ and versions this "
                f"message, START THE CONTROLLER AFTER THE uXRCE-DDS AGENT so the "
                f"version can be read off the graph.")
            return PT.TopicSpec(topic, f"px4_msgs/msg/{cls_name}", 0, cls)

        # -- callbacks ------------------------------------------------------ #
        def on_odom(self, m):
            """Convert NED/FRD -> ENU/FLU **on entry** (§9.4)."""
            p = F.ned_to_enu_vec(np.asarray(m.position, float))
            v = F.ned_to_enu_vec(np.asarray(m.velocity, float))
            q = F.px4_quat_to_enu_flu(np.asarray(m.q, float))
            # the 17-state control model needs the body rate; PX4 publishes it
            # in FRD on the same message.  Omega is NOT published and is
            # estimated inside ACMPCController.
            om = F.frd_to_flu_vec(np.asarray(m.angular_velocity, float))
            self.q_last = q
            self.state = np.concatenate([p, v, q, om])

        def on_d(self, m):
            self.d_hat = np.array([m.wrench.force.x, m.wrench.force.y,
                                   m.wrench.force.z, m.wrench.torque.x,
                                   m.wrench.torque.y, m.wrench.torque.z])

        # -- offboard handshake --------------------------------------------- #
        def _cmd(self, command, p1=0.0, p2=0.0, now=0.0):
            c = self.VehicleCommand()
            c.timestamp = int(now * 1e6)
            c.command = int(command)
            c.param1, c.param2 = float(p1), float(p2)
            c.target_system = 1
            c.target_component = 1
            c.source_system = 1
            c.source_component = 1
            c.from_external = True
            self.pub_cmd.publish(c)

        def _arm(self, now):
            """PX4 accepts OFFBOARD only after it has seen the stream (~10 msgs)."""
            self._cmd(VEHICLE_CMD_DO_SET_MODE, 1.0,
                      PX4_CUSTOM_MAIN_MODE_OFFBOARD, now)
            self._cmd(VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0, now)
            self.armed = True
            self.get_logger().info("offboard requested and arm commanded")

        def tick(self):
            if self.state is None:
                return
            now = self.get_clock().now().nanoseconds * 1e-9
            self.t0 = self.t0 if self.t0 is not None else now
            t = now - self.t0
            g = lambda k: self.get_parameter(k).value
            N = int(g("horizon"))
            dt = 1.0 / float(g("rate_hz"))

            # (p, v, a) of the flown path at an arbitrary time -- refgen takes
            # the derivatives it needs (om_ref, om_dot_ref, Omega_ref) from this
            # by the same centred differences the study's ref_state uses, so the
            # node and the study cannot drift apart on the reference.
            if self.ref_mode == "hold":
                ph = self.p_hold

                def pva(tau, _p=ph):
                    return _p, np.zeros(3), np.zeros(3)
            else:
                def pva(tau, _g=g):
                    p, v, a = L.lissajous(tau, _g("A"), _g("B"), _g("omega"),
                                          _g("z0"))
                    return p[0], v[0], a[0]

            xr, ur = RG.ref_traj(pva, t, N, dt)
            u, info = self.ctrl.step(self.state, xr, ur, d_hat=self.d_hat)

            # OffboardControlMode must stream BEFORE and DURING offboard
            mm = self.OffboardMode()
            mm.timestamp = int(now * 1e6)
            mm.body_rate = True
            self.pub_mode.publish(mm)

            om = F.ctbr_to_px4_rates(u[1:4] * np.array([10.0, 10.0, 4.0]))
            r = self.RatesSetpoint()
            r.timestamp = int(now * 1e6)
            r.roll, r.pitch, r.yaw = float(om[0]), float(om[1]), float(om[2])
            r.thrust_body = [0.0, 0.0, float(-u[0])]       # FRD, down-positive
            self.pub_rates.publish(r)

            aux = Float64MultiArray()
            aux.data = [float(x) for x in
                        list(u) + list(ur[0]) + list(np.asarray(pva(t)[0], float))]
            self.pub_aux.publish(aux)

            self.n_tick += 1
            if self.auto_arm and not self.armed and t >= self.arm_after:
                self._arm(now)
            if self.n_tick % 250 == 0:
                sm = np.asarray(self.ctrl.solve_ms[-250:])
                self.get_logger().info(
                    f"{self.cname} t={t:6.2f}s  solve p50 {np.percentile(sm,50):.2f} "
                    f"p99 {np.percentile(sm,99):.2f} ms  |d_hat| "
                    f"{np.linalg.norm(self.d_hat):.3f}")

    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
