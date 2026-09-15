"""ACMPC controller node (§9.11).

One 50 Hz tick: read state, update the causal history, run RDP inference
(watchdog armed), optionally filter, convert through B_d, solve, publish the PX4
setpoint, log timing.  Frame conversion happens at this boundary and nowhere
else: everything inside the package is ENU/FLU.
"""
from __future__ import annotations

import sys

import numpy as np

from . import frames as F
from .controller import ACMPCController


def main(args=None):                                       # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from px4_msgs.msg import (OffboardControlMode, VehicleOdometry,
                                  VehicleRatesSetpoint)
        from geometry_msgs.msg import WrenchStamped
    except Exception as ex:                               # noqa: BLE001
        print(f"acmpc_controller: no ROS 2 / px4_msgs environment "
              f"({type(ex).__name__}).  The control core is importable and "
              f"testable without one; see ACMPCController.", file=sys.stderr)
        return 1
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from px4_msgs.msg import (OffboardControlMode, VehicleOdometry,
                              VehicleRatesSetpoint)
    from geometry_msgs.msg import WrenchStamped

    sys.path.insert(0, "src/reference_generator")
    from reference_generator import lissajous as L

    class ControllerNode(Node):
        def __init__(self):
            super().__init__("acmpc_controller")
            for k, v in (("mode", "acmpc"), ("horizon", 10), ("n_iter", 10),
                         ("rate_hz", 50.0), ("use_d", True), ("A", 1.2),
                         ("B", 0.9), ("omega", 1.0), ("z0", 1.5)):
                self.declare_parameter(k, v)
            g = lambda k: self.get_parameter(k).value
            ok, pv, pa, budget = L.check_feasible(g("A"), g("B"), g("omega"))
            if not ok:
                raise SystemExit(
                    f"reference is INFEASIBLE: peak demand {pa:.3f} m/s^2 "
                    f"exceeds alpha*a_lat_max = {budget:.4f}.  Refusing to fly "
                    f"it; a benchmark built on an infeasible reference measures "
                    f"the reference, not the controller.")
            self.get_logger().info(f"reference feasible: peak_a {pa:.3f} <= "
                                   f"{budget:.4f} m/s^2")
            self.ctrl = ACMPCController(mode=g("mode"), horizon=int(g("horizon")),
                                        n_iter=int(g("n_iter")), use_d=g("use_d"))
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
            self.state = None
            self.d_hat = np.zeros(6)
            self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry",
                                     self.on_odom, qos)
            self.create_subscription(WrenchStamped, "/rdp/disturbance_estimate",
                                     self.on_d, 10)
            self.pub_rates = self.create_publisher(
                VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", 10)
            self.pub_mode = self.create_publisher(
                OffboardControlMode, "/fmu/in/offboard_control_mode", 10)
            self.t0 = None
            self.create_timer(1.0 / float(g("rate_hz")), self.tick)

        def on_odom(self, m):
            """Convert NED/FRD -> ENU/FLU **on entry** (§9.4)."""
            p = F.ned_to_enu_vec(np.asarray(m.position, float))
            v = F.ned_to_enu_vec(np.asarray(m.velocity, float))
            q = F.px4_quat_to_enu_flu(np.asarray(m.q, float))
            self.state = np.concatenate([p, v, q])

        def on_d(self, m):
            self.d_hat = np.array([m.wrench.force.x, m.wrench.force.y,
                                   m.wrench.force.z, m.wrench.torque.x,
                                   m.wrench.torque.y, m.wrench.torque.z])

        def tick(self):
            if self.state is None:
                return
            now = self.get_clock().now().nanoseconds * 1e-9
            self.t0 = self.t0 if self.t0 is not None else now
            t = now - self.t0
            g = lambda k: self.get_parameter(k).value
            N = int(g("horizon"))
            dt = 1.0 / float(g("rate_hz"))
            xr, ur = [], []
            for k in range(N + 1):
                p, v, a = L.lissajous(t + k * dt, g("A"), g("B"), g("omega"), g("z0"))
                zb = a[0] + np.array([0, 0, 9.8066])
                nz = np.linalg.norm(zb)
                ax = np.cross([0, 0, 1.0], zb / nz)
                ang = np.arccos(np.clip(zb[2] / nz, -1, 1))
                s = np.linalg.norm(ax)
                qr = (np.array([1.0, 0, 0, 0]) if s < 1e-12 else
                      np.concatenate([[np.cos(ang / 2)], ax / s * np.sin(ang / 2)]))
                xr.append(np.concatenate([p[0], v[0], qr]))
                if k < N:
                    ur.append(np.array([np.sqrt(min(2.0643076923076924 * nz
                                                    / 34.19432, 1.0)), 0, 0, 0]))
            u, info = self.ctrl.step(self.state, np.array(xr), np.array(ur),
                                     d_hat=self.d_hat)
            om = F.ctbr_to_px4_rates(u[1:4] * np.array([10.0, 10.0, 4.0]))
            mm = OffboardControlMode()
            mm.timestamp = int(now * 1e6); mm.body_rate = True
            self.pub_mode.publish(mm)
            r = VehicleRatesSetpoint()
            r.timestamp = int(now * 1e6)
            r.roll, r.pitch, r.yaw = float(om[0]), float(om[1]), float(om[2])
            r.thrust_body = [0.0, 0.0, float(-u[0])]       # FRD, down-positive
            self.pub_rates.publish(r)

    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
