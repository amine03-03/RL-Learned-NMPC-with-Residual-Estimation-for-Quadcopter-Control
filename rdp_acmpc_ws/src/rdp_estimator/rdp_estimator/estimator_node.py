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
        # rdp_infer lives in the repository's study/, outside this workspace.
        from acmpc_controller.px4_topics import add_study_to_path
        add_study_to_path()
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
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from geometry_msgs.msg import WrenchStamped
        from std_msgs.msg import Float64MultiArray
    except Exception as ex:                               # noqa: BLE001
        print(f"rdp_estimator: no ROS 2 environment ({type(ex).__name__}). "
              f"The core is importable and testable without one; see "
              f"EstimatorCore.", file=sys.stderr)
        return 1
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from geometry_msgs.msg import WrenchStamped
    from std_msgs.msg import Float64MultiArray
    from acmpc_controller import frames as F
    from acmpc_controller import px4_topics as PT

    class RDPNode(Node):
        """Feeds the causal buffer from state and actuator topics, and ONLY those.

        The window is assembled here, in the odometry callback, from:

            p, R, v, omega   <- /fmu/out/vehicle_odometry   (converted to ENU/FLU)
            PWM              <- /fmu/out/actuator_motors
            u_prev, u_ref,
            p_ref            <- /controller/frame_aux

        ``/disturbance/ground_truth`` is deliberately absent from that list and
        there is no code path by which it could enter (§9.1).  The buffer is
        pushed at the odometry rate rather than on a timer, so a frame is a
        genuine sample of the vehicle and not the last one repeated.
        """

        def __init__(self):
            super().__init__("rdp_estimator")
            for k, v in (("model_path", "models/rdp_gru.npz"), ("rate_hz", 50.0),
                         ("filter_alpha", 0.0), ("budget_ms", 8.0),
                         ("px4_namespace", ""), ("topic_timeout_s", 30.0),
                         ("require_actuator_motors", True)):
                self.declare_parameter(k, v)
            g = lambda k: self.get_parameter(k).value
            self.core = EstimatorCore(g("model_path"), filt=g("filter_alpha"),
                                      budget_ms=g("budget_ms"))
            ns, tmo = g("px4_namespace"), float(g("topic_timeout_s"))
            odom = PT.resolve(self, "/fmu/out/vehicle_odometry", tmo, ns)
            am = PT.resolve(self, "/fmu/out/actuator_motors", 5.0, ns,
                            required=bool(g("require_actuator_motors")))
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
            self.pwm = np.zeros(4)
            self.u_prev = np.zeros(4)
            self.u_ref = np.zeros(4)
            self.p_ref = np.zeros(3)
            self.n_push = 0
            self.create_subscription(odom.msg_class, odom.topic, self.on_odom, qos)
            self.get_logger().info(f"state from {odom}")
            if am is not None:
                self.create_subscription(am.msg_class, am.topic, self.on_motors, qos)
                self.get_logger().info(f"PWM from {am}")
            else:
                self.get_logger().error(
                    "actuator_motors is NOT publishing.  Per §6.2 the PWM block "
                    "is the only observable channel for a STANDING moment, so "
                    "the moment outputs of this estimator are unobservable "
                    "without it.  Proceeding because require_actuator_motors is "
                    "false; the moment channels should not be believed.")
            self.create_subscription(Float64MultiArray, "/controller/frame_aux",
                                     self.on_aux, 10)
            self.pub = self.create_publisher(WrenchStamped,
                                             "/rdp/disturbance_estimate", 10)
            self.create_timer(1.0 / float(g("rate_hz")), self.tick)
            self.get_logger().info(
                f"RDP up: {self.core.rdp.kind}, H={self.core.H}, "
                f"channels={self.core.rdp.channels}")

        def on_motors(self, m):
            self.pwm = np.asarray(m.control, float)[:4]

        def on_aux(self, m):
            d = np.asarray(m.data, float)
            if d.size >= 11:
                self.u_prev, self.u_ref, self.p_ref = d[0:4], d[4:8], d[8:11]

        def on_odom(self, m):
            p = F.ned_to_enu_vec(np.asarray(m.position, float))
            v = F.ned_to_enu_vec(np.asarray(m.velocity, float))
            q = F.px4_quat_to_enu_flu(np.asarray(m.q, float))
            om = F.frd_to_flu_vec(np.asarray(m.angular_velocity, float))
            R = F.qrotmat(q[None])[0]
            stamp = float(getattr(m, "timestamp_sample", m.timestamp)) * 1e-6
            self.core.push(p, self.p_ref, R, v, om, self.u_prev, self.u_ref,
                           self.pwm, stamp=stamp)
            self.n_push += 1
            if self.n_push == self.core.H:
                self.get_logger().info(
                    f"causal window full after {self.n_push} frames "
                    f"({self.core.buf.span_s():.2f} s); estimates are live")

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
    except KeyboardInterrupt:          # launch sends SIGINT; that is not a fault
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
