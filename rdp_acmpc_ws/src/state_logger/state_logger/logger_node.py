"""50 Hz synchronised logger node (§9.6).  Writes the CSV of `logger.py`."""
from __future__ import annotations

import sys


def main(args=None):                                       # pragma: no cover
    try:
        import rclpy
    except Exception as ex:                               # noqa: BLE001
        print(f"state_logger: no ROS 2 environment ({type(ex).__name__}). "
              f"StateLogger is importable without one; see logger.py.",
              file=sys.stderr)
        return 1
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import WrenchStamped
    from .logger import StateLogger

    class LoggerNode(Node):
        def __init__(self):
            super().__init__("state_logger")
            self.declare_parameter("run_dir", "runs/experiment_0")
            self.declare_parameter("rate_hz", 50.0)
            self.log = StateLogger(self.get_parameter("run_dir").value)
            self.d_true = np.zeros(6)
            self.d_pred = np.zeros(6)
            self.create_subscription(WrenchStamped, "/disturbance/ground_truth",
                                     lambda m: self._set("d_true", m), 10)
            self.create_subscription(WrenchStamped, "/rdp/disturbance_estimate",
                                     lambda m: self._set("d_pred", m), 10)
            self.t0 = None
            self.create_timer(1.0 / float(self.get_parameter("rate_hz").value),
                              self.tick)

        def _set(self, k, m):
            setattr(self, k, np.array([m.wrench.force.x, m.wrench.force.y,
                                       m.wrench.force.z, m.wrench.torque.x,
                                       m.wrench.torque.y, m.wrench.torque.z]))

        def tick(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            self.t0 = self.t0 if self.t0 is not None else now
            t = now - self.t0
            z3 = np.zeros(3)
            self.log.log(t, 0, "nominal", z3, z3, np.array([1.0, 0, 0, 0]), z3,
                         z3, z3, np.zeros(4), np.zeros(4), self.d_true, self.d_pred)

        def destroy_node(self):
            self.log.close()
            super().destroy_node()

    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
