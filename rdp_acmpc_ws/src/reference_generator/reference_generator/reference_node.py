"""Reference generator node (§9.9).

Publishes (9.5) and **refuses to start** if the requested aggressiveness
violates the feasibility envelope (2.6): a benchmark built on an infeasible
reference measures the reference, not the controller.
"""
from __future__ import annotations

import sys

import numpy as np

from .lissajous import LEVELS, check_feasible, episode_timeline, lissajous


def main(args=None):                                       # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from nav_msgs.msg import Path
        from geometry_msgs.msg import PoseStamped
    except Exception as ex:                               # noqa: BLE001
        print(f"reference_generator: no ROS 2 environment ({type(ex).__name__}). "
              f"The trajectory and its feasibility gate are importable without "
              f"one; see lissajous.py.", file=sys.stderr)
        return 1
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped

    class RefNode(Node):
        def __init__(self):
            super().__init__("reference_generator")
            for k, v in (("A", 1.2), ("B", 0.9), ("level", "moderate"),
                         ("z0", 1.5), ("rate_hz", 50.0)):
                self.declare_parameter(k, v)
            g = lambda k: self.get_parameter(k).value
            w = LEVELS.get(g("level"), 1.0)
            ok, pv, pa, budget = check_feasible(g("A"), g("B"), w)
            if not ok:
                raise SystemExit(
                    f"level {g('level')!r} (w={w}) is INFEASIBLE: peak demand "
                    f"{pa:.3f} m/s^2 > alpha*a_lat_max = {budget:.4f}.  "
                    f"Refusing to publish it.")
            self.w = w
            self.get_logger().info(
                f"level {g('level')} w={w}: peak_v {pv:.3f} m/s, peak_a "
                f"{pa:.3f} m/s^2 <= {budget:.4f} -- feasible")
            self.pub = self.create_publisher(PoseStamped, "/reference/trajectory", 10)
            self.t0 = None
            self.create_timer(1.0 / float(g("rate_hz")), self.tick)

        def tick(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            self.t0 = self.t0 if self.t0 is not None else now
            t = now - self.t0
            g = lambda k: self.get_parameter(k).value
            p, _, _ = lissajous(t, g("A"), g("B"), self.w, g("z0"))
            m = PoseStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = episode_timeline(t)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p[0]
            self.pub.publish(m)

    rclpy.init(args=args)
    node = RefNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
