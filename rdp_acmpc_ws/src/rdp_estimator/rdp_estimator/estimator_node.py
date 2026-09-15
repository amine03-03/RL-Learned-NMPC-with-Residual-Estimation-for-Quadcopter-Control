"""RDP estimator node (§9.5, §9.11 steps 2-4).

Subscribes state and actuator commands, maintains the causal ring buffer, runs
the **pure NumPy** ``rdp_infer`` forward pass under a watchdog, optionally
filters, and publishes ``/rdp/disturbance_estimate``.

It never subscribes to ``/disturbance/ground_truth``.  That is not a convention
here but the structure: the buffer is fed only from state and actuator topics,
so there is no path by which an injected value could enter a window (§9.1).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from .ring_buffer import RingBuffer, build_frame
from .watchdog import CausalFilter, TimingRecorder, Watchdog


class EstimatorCore:
    """ROS-free core, so the whole data path can be tested without a graph."""

    def __init__(self, model_path, H=None, filt=0.0, budget_ms=8.0):
        import rdp_infer
        self.rdp = rdp_infer.load(model_path)
        self.H = int(H or self.rdp.H)
        if self.H != self.rdp.H:
            raise ValueError(f"requested H={self.H} but the model was exported "
                             f"with H={self.rdp.H}; re-export rather than "
                             f"reshaping the window")
        self.buf = RingBuffer(self.H, self.rdp.frame_dim)
        self.wd = Watchdog(budget_ms=budget_ms, out_dim=6)
        self.filt = CausalFilter(filt, 6)
        self.timing = TimingRecorder()
        self.last = np.zeros(6)

    def push(self, p, p_ref, R, v, om, u_prev, u_ref, pwm, stamp=0.0):
        self.buf.push(build_frame(p, p_ref, R, v, om, u_prev, u_ref, pwm), stamp)

    def estimate(self):
        """-> (d_hat (6,), info).  Zero until the window is full."""
        if not self.buf.ready():
            self.timing.add("rdp", 0.0)
            return np.zeros(6), dict(ready=False, fault=False, ms=0.0)
        y, ms, fault = self.wd(self.rdp.predict, self.buf.window())
        y = np.asarray(y).reshape(-1)
        y = self.filt(y) if not fault else y
        self.timing.add("rdp", ms)
        self.last = y
        return y, dict(ready=True, fault=fault, ms=ms)

    def stats(self):
        return dict(watchdog=self.wd.stats(), timing=self.timing.summary())


def _degrade(y, delay_buf, delay_steps=0, noise_sd=0.0, scale=1.0, rng=None):
    """C3: RDP with injected delay, additive noise and bounded scale error."""
    rng = rng or np.random.default_rng(0)
    delay_buf.append(np.asarray(y, float).copy())
    if len(delay_buf) > delay_steps + 1:
        delay_buf.pop(0)
    out = delay_buf[0] * scale
    if noise_sd > 0:
        out = out + rng.normal(0.0, noise_sd, size=out.shape)
    return out


def main(args=None):                                       # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import WrenchStamped
    except Exception as ex:                               # noqa: BLE001
        print(f"rdp_estimator: no ROS 2 environment ({type(ex).__name__}). "
              f"The core is importable and testable without one; see "
              f"EstimatorCore.", file=sys.stderr)
        return 1
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import WrenchStamped

    class RDPNode(Node):
        def __init__(self):
            super().__init__("rdp_estimator")
            self.declare_parameter("model_path", "models/rdp_gru.npz")
            self.declare_parameter("rate_hz", 50.0)
            self.declare_parameter("filter_alpha", 0.0)
            self.declare_parameter("budget_ms", 8.0)
            mp = self.get_parameter("model_path").value
            self.core = EstimatorCore(mp,
                                      filt=self.get_parameter("filter_alpha").value,
                                      budget_ms=self.get_parameter("budget_ms").value)
            self.pub = self.create_publisher(WrenchStamped,
                                             "/rdp/disturbance_estimate", 10)
            hz = float(self.get_parameter("rate_hz").value)
            self.create_timer(1.0 / hz, self.tick)
            self.get_logger().info(
                f"RDP up: {self.core.rdp.kind}, H={self.core.H}, "
                f"channels={self.core.rdp.channels}")

        def tick(self):
            d, info = self.core.estimate()
            m = WrenchStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "map"
            m.wrench.force.x, m.wrench.force.y, m.wrench.force.z = d[:3]
            m.wrench.torque.x, m.wrench.torque.y, m.wrench.torque.z = d[3:]
            self.pub.publish(m)
            if info["fault"]:
                self.get_logger().warn(
                    f"RDP inference fault -> falling back to zero "
                    f"({self.core.wd.stats()['last_fault']})")

    rclpy.init(args=args)
    node = RDPNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
