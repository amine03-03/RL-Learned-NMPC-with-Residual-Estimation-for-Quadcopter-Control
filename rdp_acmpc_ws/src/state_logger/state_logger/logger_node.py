"""50 Hz synchronised logger node (§9.6).  Writes the CSV of ``logger.py``.

One row per ``/acmpc/status`` message, i.e. per controller tick, so state,
reference and command in a row belong to the same solve.  The other channels
are the latest sample received before that tick.

Subscribes  /acmpc/status               state, reference, command (status_msg)
            /fmu/out/actuator_motors    PWM block
            /disturbance/ground_truth   true wrench (evaluator channel)
            /disturbance/phase          nominal | disturbed | recovery
            /rdp/disturbance_estimate   predicted wrench
            /rdp/status                 [inference ms, fault, ready]
Writes      <run_dir>/states.csv
"""
from __future__ import annotations

import sys

import numpy as np


def main(args=None):                                       # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from geometry_msgs.msg import WrenchStamped
        from std_msgs.msg import Float64MultiArray, String
        from px4_msgs.msg import ActuatorMotors
    except Exception as ex:                               # noqa: BLE001
        print(f"state_logger: no ROS 2 / px4_msgs environment ({type(ex).__name__}). "
              f"StateLogger is importable without one; see logger.py.",
              file=sys.stderr)
        return 1
    from acmpc_controller import status_msg
    from .logger import StateLogger

    def wrench6(m):
        return np.array([m.wrench.force.x, m.wrench.force.y, m.wrench.force.z,
                         m.wrench.torque.x, m.wrench.torque.y, m.wrench.torque.z])

    class LoggerNode(Node):
        def __init__(self):
            super().__init__("state_logger")
            self.declare_parameter("run_dir", "runs/experiment_0")
            self.declare_parameter("episode", 0)
            self.log = StateLogger(self.get_parameter("run_dir").value)
            self.episode = int(self.get_parameter("episode").value)
            self.d_true = np.zeros(6)
            self.d_pred = np.zeros(6)
            self.pwm = np.full(4, np.nan)
            self.phase = "unknown"
            self.rdp = (np.nan, 0.0, 0.0)
            qos_px4 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
            sub = self.create_subscription
            sub(Float64MultiArray, "/acmpc/status", self.on_status, 10)
            sub(ActuatorMotors, "/fmu/out/actuator_motors",
                lambda m: setattr(self, "pwm", np.asarray(m.control[0:4], float)), qos_px4)
            sub(WrenchStamped, "/disturbance/ground_truth",
                lambda m: setattr(self, "d_true", wrench6(m)), 10)
            sub(WrenchStamped, "/rdp/disturbance_estimate",
                lambda m: setattr(self, "d_pred", wrench6(m)), 10)
            sub(String, "/disturbance/phase",
                lambda m: setattr(self, "phase", m.data), 10)
            sub(Float64MultiArray, "/rdp/status",
                lambda m: setattr(self, "rdp", tuple(m.data)), 10)
            self.get_logger().info(f"logging to {self.log.path}")

        def on_status(self, m):
            try:
                self._on_status(m)
            except Exception:                             # noqa: BLE001
                if rclpy.ok():                            # a real error: surface it
                    raise                                 # (else Ctrl-C mid-callback)

        def _on_status(self, m):
            st = status_msg.unpack(m.data)
            ms, fault, ready = self.rdp
            self.log.log(st["t"], self.episode, self.phase, st["p"], st["v"], st["q"],
                         st["om"], st["p_ref"], st["v_ref"], st["u"], self.pwm,
                         self.d_true, self.d_pred, rdp_ms=float(ms),
                         acmpc_ms=st["solve_ms"], loop_ms=st["loop_ms"],
                         rdp_fault=bool(fault),
                         rdp_ready=bool(ready))

        def destroy_node(self):
            self.log.close()
            super().destroy_node()

    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=args)
    node = LoggerNode()
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
