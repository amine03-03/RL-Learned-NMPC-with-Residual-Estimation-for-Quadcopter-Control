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
        from geometry_msgs.msg import WrenchStamped
    except Exception as ex:                               # noqa: BLE001
        print(f"disturbance_manager: no ROS 2 environment ({type(ex).__name__}). "
              f"Scenario definitions are importable without one; see scenarios.py.",
              file=sys.stderr)
        return 1
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import WrenchStamped

    class ManagerNode(Node):
        def __init__(self):
            super().__init__("disturbance_manager")
            self.declare_parameter("scenario", "S0")
            self.declare_parameter("seed", 0)
            self.declare_parameter("rate_hz", 50.0)
            self.declare_parameter("t_on", 5.0)
            self.declare_parameter("t_off", 12.0)
            g = lambda k: self.get_parameter(k).value
            self.scen = Scenario(g("scenario"), seed=int(g("seed")),
                                 t_on=float(g("t_on")), t_off=float(g("t_off")))
            self.get_logger().info(f"scenario {self.scen.describe()}")
            self.pub = self.create_publisher(WrenchStamped,
                                             "/disturbance/ground_truth", 10)
            self.t0 = None
            self.create_timer(1.0 / float(g("rate_hz")), self.tick)

        def tick(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            self.t0 = self.t0 if self.t0 is not None else now
            F, tau = self.scen.wrench(now - self.t0)
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
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
