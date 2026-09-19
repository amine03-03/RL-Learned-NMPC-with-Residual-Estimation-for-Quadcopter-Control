"""Disturbance manager node (§9.7).

Applies S0..S6 to the simulated plant and publishes the **ground truth** on
``/disturbance/ground_truth``.  That topic is for the evaluator and the logger
only; §9.1 forbids it reaching the online controller, and the RDP subscribes to
state and actuator topics exclusively, so there is no path for it to.
"""
from __future__ import annotations

import sys

import numpy as np

from .scenarios import Scenario


def main(args=None):                                       # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from geometry_msgs.msg import WrenchStamped
    except Exception as ex:                               # noqa: BLE001
        print(f"disturbance_manager: no ROS 2 environment ({type(ex).__name__}). "
              f"Scenario definitions are importable without one; see scenarios.py.",
              file=sys.stderr)
        return 1
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import WrenchStamped

    class ManagerNode(Node):
        """Publishes the scenario's true wrench for the logger and the plots.

        S7 is different from S0..S6 in two ways and both are handled here.  The
        payload is *bolted on*, not switched on at ``t_on``, and it is attached
        to the Gazebo model by ``tools/payload_sdf.py`` -- this node does not
        apply it, it reports it.  And its moment is taken in the body frame,
        so the truth depends on the current attitude, which is why S7 (and only
        S7) subscribes to odometry.
        """

        def __init__(self):
            super().__init__("disturbance_manager")
            for k, v in (("scenario", "S0"), ("seed", 0), ("rate_hz", 50.0),
                         ("t_on", 5.0), ("t_off", 12.0),
                         ("payload_mass", 0.0), ("payload_offset", [0.0, 0.0, 0.0]),
                         ("px4_namespace", ""), ("topic_timeout_s", 30.0)):
                self.declare_parameter(k, v)
            g = lambda k: self.get_parameter(k).value
            params = {}
            if float(g("payload_mass")) > 0.0:
                from acmpc_controller import px4_topics as _PT
                params = dict(
                    payload_mass=float(g("payload_mass")),
                    payload_offset=_PT.as_floats(
                        g("payload_offset"), 3, "payload_offset").tolist())
            self.scen = Scenario(g("scenario"), seed=int(g("seed")),
                                 t_on=float(g("t_on")), t_off=float(g("t_off")),
                                 params=params)
            self.get_logger().info(f"scenario {self.scen.describe()}")
            self.q = None
            if self.scen.sid == "S7":
                from acmpc_controller import frames as F
                from acmpc_controller import px4_topics as PT
                self._F = F
                odom = PT.resolve(self, "/fmu/out/vehicle_odometry",
                                  float(g("topic_timeout_s")), g("px4_namespace"),
                                  required=False)
                if odom is not None:
                    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                     history=HistoryPolicy.KEEP_LAST, depth=5)
                    self.create_subscription(odom.msg_class, odom.topic,
                                             self.on_odom, qos)
                    self.get_logger().info(f"S7 attitude from {odom}")
                else:
                    self.get_logger().warn(
                        "S7 without odometry: the published moment assumes hover "
                        "(R = I).  Exact to first order at hover tilts, and the "
                        "plots say so.")
                F0, tau0 = self.scen.payload_wrench()
                self.get_logger().info(
                    f"S7 payload: m_p={self.scen.params['payload_mass']:.4f} kg "
                    f"at {self.scen.params['payload_offset']}; hover truth "
                    f"F_z={F0[2]:+.4f} N  tau=({tau0[0]:+.4f}, {tau0[1]:+.4f}, "
                    f"{tau0[2]:+.4f}) N m.  The PLANT-side payload is attached by "
                    f"tools/payload_sdf.py -- check it is installed.")
            self.pub = self.create_publisher(WrenchStamped,
                                             "/disturbance/ground_truth", 10)
            self.t0 = None
            self.create_timer(1.0 / float(g("rate_hz")), self.tick)

        def on_odom(self, m):
            self.q = self._F.px4_quat_to_enu_flu(np.asarray(m.q, float))

        def tick(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            self.t0 = self.t0 if self.t0 is not None else now
            F, tau = self.scen.wrench(now - self.t0, self.q)
            m = WrenchStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "map"          # force is world (ENU), torque body
            m.wrench.force.x, m.wrench.force.y, m.wrench.force.z = F
            m.wrench.torque.x, m.wrench.torque.y, m.wrench.torque.z = tau
            self.pub.publish(m)

    rclpy.init(args=args)
    node = ManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:          # launch sends SIGINT; that is not a fault
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
