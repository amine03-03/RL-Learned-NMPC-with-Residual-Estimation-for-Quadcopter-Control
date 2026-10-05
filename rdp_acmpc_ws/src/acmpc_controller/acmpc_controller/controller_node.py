"""ACMPC controller node (§9.11).

One 50 Hz tick: read the latest PX4 odometry, build the reference and the
observation with the study's functions, solve, publish the PX4 rate setpoint,
and publish ``/acmpc/status`` (state, reference, command of this tick) for the
estimator and the logger.  Frame conversion happens at this boundary and
nowhere else: everything inside the package is ENU/FLU.

Subscribes  /fmu/out/vehicle_odometry      px4_msgs/VehicleOdometry
            /fmu/out/vehicle_status_v1     px4_msgs/VehicleStatus (nav_state)
            /rdp/disturbance_estimate      geometry_msgs/WrenchStamped
Publishes   /fmu/in/offboard_control_mode  px4_msgs/OffboardControlMode
            /fmu/in/vehicle_rates_setpoint px4_msgs/VehicleRatesSetpoint
            /acmpc/status                  std_msgs/Float64MultiArray (status_msg)

Hand-over.  Until PX4 reports OFFBOARD, the setpoints PX4 needs before it
accepts the switch hold the vehicle's current position.  At OFFBOARD the
reference clock starts with an *entry* trajectory (reference.entry_reference):
a minimum-jerk transfer from the measured position and velocity to the path's
start state, then the path itself.  Without it the hand-over is a metre-scale
step that the NMPC (tuned on episodes starting on the path) answers with zero
collective -- measured in PX4 SITL, the vehicle flipped.  PX4 v1.16
publishes VehicleStatus (MESSAGE_VERSION 1) as ``vehicle_status_v1``.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from . import frames as F
from . import status_msg
from .controller import ACMPCController
from .reference import entry_reference, hold_at


def _lissajous():
    try:
        from reference_generator import lissajous as L
    except ImportError:                  # source tree without a colcon install
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "reference_generator"))
        from reference_generator import lissajous as L
    return L


def make_reference(level, A, B, z0):
    """-> (pva(t), hold).  Refuses an infeasible reference."""
    L = _lissajous()
    if level != L.HOLD and level not in L.LEVELS:
        raise ValueError(f"level must be one of {sorted(L.LEVELS)} or 'hold', "
                         f"got {level!r}")
    ok, _, pa, budget = L.check_feasible(A, B, L.LEVELS.get(level, 0.0), mode=level)
    if not ok:
        raise SystemExit(
            f"reference {level!r} is INFEASIBLE: peak demand {pa:.3f} m/s^2 exceeds "
            f"alpha*a_lat_max = {budget:.4f}.  Refusing to fly it; a benchmark built "
            f"on an infeasible reference measures the reference, not the controller.")
    return (lambda t: L.reference(t, level, A, B, z0)), level == L.HOLD


def main(args=None):                                       # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from px4_msgs.msg import (OffboardControlMode, VehicleOdometry,
                                  VehicleRatesSetpoint, VehicleStatus)
        from geometry_msgs.msg import WrenchStamped
        from std_msgs.msg import Float64MultiArray
    except Exception as ex:                               # noqa: BLE001
        print(f"acmpc_controller: no ROS 2 / px4_msgs environment "
              f"({type(ex).__name__}).  The control core is importable and "
              f"testable without one; see ACMPCController.", file=sys.stderr)
        return 1

    class ControllerNode(Node):
        def __init__(self):
            super().__init__("acmpc_controller")
            for k, v in (("mode", "nmpc1"), ("horizon", 10), ("n_iter", 10),
                         ("rate_hz", 50.0), ("use_d", True),
                         ("dmod_mode", "closed_loop"), ("level", "moderate"),
                         ("A", 1.2), ("B", 0.9), ("z0", 1.5),
                         ("odom_timeout_s", 0.5), ("start_on_offboard", True),
                         ("entry_s", 4.0), ("entry_a_max", 3.0), ("delay_steps", 1),
                         ("status_topic", "/fmu/out/vehicle_status_v1")):
                self.declare_parameter(k, v)
            g = lambda k: self.get_parameter(k).value
            self.pva, self.hold = make_reference(g("level"), float(g("A")),
                                                 float(g("B")), float(g("z0")))
            self.ctrl = ACMPCController(mode=g("mode"), horizon=int(g("horizon")),
                                        n_iter=int(g("n_iter")), use_d=g("use_d"),
                                        dmod_mode=g("dmod_mode"),
                                        delay_steps=int(g("delay_steps")))
            self._warm_up()
            self.get_logger().info(
                f"mode={g('mode')} N={g('horizon')} reference={g('level')} "
                f"(A={g('A')}, B={g('B')}, z0={g('z0')})  oracle_obs={self.ctrl.oracle}  "
                f"delay_steps={self.ctrl.delay_steps}")

            qos_px4 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
            self.odom = None
            self.odom_stamp = None
            self.odom_timeout_us = int(1e6 * float(g("odom_timeout_s")))
            self.d_hat = None
            self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry",
                                     self.on_odom, qos_px4)
            self.create_subscription(WrenchStamped, "/rdp/disturbance_estimate",
                                     self.on_d, 10)
            self.start_on_offboard = bool(g("start_on_offboard"))
            self.offboard = False
            self.create_subscription(VehicleStatus, g("status_topic"),
                                     self.on_status_px4, qos_px4)
            self.pub_rates = self.create_publisher(
                VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", qos_px4)
            self.pub_mode = self.create_publisher(
                OffboardControlMode, "/fmu/in/offboard_control_mode", qos_px4)
            self.pub_status = self.create_publisher(Float64MultiArray,
                                                    "/acmpc/status", 10)
            self.t0 = None
            self.create_timer(1.0 / float(g("rate_hz")), self.tick)

        def _warm_up(self):
            """JIT-compile the solve before the first real tick, so the first
            setpoints are not late enough to trip PX4's offboard-loss timeout."""
            p0 = np.asarray(self.pva(np.zeros(1))[0])[0]
            x10 = np.concatenate([p0, np.zeros(3), [1.0, 0, 0, 0]])
            self.ctrl.set_reference(self.pva, False)
            for _ in range(2):
                self.ctrl.tick(x10, np.zeros(3), 0.0)
            self.ctrl.set_reference(self.pva, False)          # reset the running state
            self.ctrl.last_u = np.array([self.ctrl.X.U_HOVER, 0.0, 0.0, 0.0])

        def on_odom(self, m):
            """NED/FRD -> ENU/FLU **on entry** (§9.4)."""
            self.odom = F.odometry_to_enu_flu(m.position, m.q, m.velocity,
                                              m.angular_velocity, m.velocity_frame)
            self.odom_stamp = self.get_clock().now().nanoseconds // 1000

        def on_status_px4(self, m):
            self.offboard = m.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD

        def on_d(self, m):
            self.d_hat = np.array([m.wrench.force.x, m.wrench.force.y,
                                   m.wrench.force.z, m.wrench.torque.x,
                                   m.wrench.torque.y, m.wrench.torque.z])

        def tick(self):
            try:
                self._tick()
            except Exception:                             # noqa: BLE001
                if rclpy.ok():                            # a real error: surface it
                    raise                                 # (else Ctrl-C mid-tick)

        def _tick(self):
            if self.odom is None:
                return
            t_start = time.perf_counter()
            now_us = self.get_clock().now().nanoseconds // 1000
            if now_us - self.odom_stamp > self.odom_timeout_us:
                # stale state: stop streaming, so PX4's offboard-loss failsafe
                # (COM_OF_LOSS_T) takes over instead of us flying blind
                self.get_logger().error("odometry stale -- not publishing setpoints",
                                        throttle_duration_sec=1.0)
                return
            p, v, q, om = self.odom
            g = lambda k: self.get_parameter(k).value
            if self.t0 is None and (self.offboard or not self.start_on_offboard):
                pva, T = entry_reference(self.pva, p, v, float(g("entry_s")),
                                         float(g("entry_a_max")))
                self.ctrl.set_reference(pva, False)
                self.t0 = now_us
                self.get_logger().info(f"reference clock started: {T:.1f} s entry "
                                       f"from ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) m, "
                                       f"then the path")
            if self.t0 is None:
                self.ctrl.set_reference(hold_at(p), False)     # benign until OFFBOARD
                self.get_logger().info("holding position; waiting for PX4 OFFBOARD",
                                       throttle_duration_sec=10.0)
            t = 0.0 if self.t0 is None else (now_us - self.t0) * 1e-6
            x10 = np.concatenate([p, v, q])
            u, info = self.ctrl.tick(x10, om, t, d_hat=self.d_hat)

            mm = OffboardControlMode()
            mm.timestamp = int(now_us)
            mm.position = mm.velocity = mm.acceleration = mm.attitude = False
            mm.body_rate = True
            self.pub_mode.publish(mm)

            om_px4 = F.ctbr_to_px4_rates(u[1:4] * np.asarray(self.ctrl.X.OM_MAX))
            r = VehicleRatesSetpoint()
            r.timestamp = int(now_us)
            r.roll, r.pitch, r.yaw = (float(z) for z in om_px4)
            r.thrust_body = [0.0, 0.0, float(-u[0])]       # FRD, down-positive
            self.pub_rates.publish(r)

            xr = info["x_ref"]
            st = Float64MultiArray()
            st.data = status_msg.pack(
                t=t, p=p, v=v, q=q, om=om, p_ref=xr[0:3], v_ref=xr[3:6],
                q_ref=xr[6:10], u_prev=info["u_prev"], u_ref=info["u_ref"], u=u,
                solve_ms=info["solve_ms"],
                loop_ms=1e3 * (time.perf_counter() - t_start))
            self.pub_status.publish(st)

    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:                                     # noqa: BLE001
        if rclpy.ok():                                    # not a shutdown race
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
