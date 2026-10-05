"""RDP estimator node (§9.5, §9.11 steps 2-4).

Subscribes state and actuator commands, maintains the causal ring buffer, runs
the **pure NumPy** ``rdp_infer`` forward pass under a watchdog, smooths and
optionally filters, and publishes ``/rdp/disturbance_estimate``.

Subscribes  /acmpc/status               std_msgs/Float64MultiArray (state, p_ref,
                                        u_{t-1}, u_ref of one controller tick)
            /fmu/out/actuator_motors    px4_msgs/ActuatorMotors (the PWM block)
Publishes   /rdp/disturbance_estimate   geometry_msgs/WrenchStamped [N world, N.m body]
            /rdp/status                 std_msgs/Float64MultiArray [ms, fault, ready]

One frame is pushed per ``/acmpc/status`` message, i.e. at the controller's
50 Hz and on its clock, so ``p - p_ref`` and ``u_{t-1} - u_ref`` are evaluated
exactly as in the study's ``_frame26``.

It never subscribes to ``/disturbance/ground_truth``.  That is not a convention
here but the structure: the buffer is fed only from state and actuator topics,
so there is no path by which an injected value could enter a window (§9.1).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from .ring_buffer import RingBuffer, build_frame, quat_to_R
from .watchdog import CausalFilter, TimingRecorder, Watchdog


class EstimatorCore:
    """ROS-free core, so the whole data path can be tested without a graph."""

    def __init__(self, model_path, H=None, filt=0.0, budget_ms=8.0, smooth_n=1):
        try:
            import rdp_infer
        except ImportError:              # study/ not on PYTHONPATH: resolve it
            from acmpc_controller import _study_path  # noqa: F401  (path only, no JAX)
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
        # arXiv:2605.16015 / AdaptEnv.attach: the prediction that reaches the
        # controller is the MEAN over a rolling buffer of smooth_n predictions
        self.smooth_n = max(int(smooth_n), 1)
        self._hist = []

    def push(self, p, p_ref, R, v, om, u_prev, u_ref, pwm, stamp=0.0):
        self.buf.push(build_frame(p, p_ref, R, v, om, u_prev, u_ref, pwm), stamp)

    def push_status(self, st, pwm):
        """One frame (6.2) from an unpacked ``/acmpc/status`` and the PWM block."""
        self.push(st["p"], st["p_ref"], quat_to_R(st["q"]), st["v"], st["om"],
                  st["u_prev"], st["u_ref"], pwm, stamp=st["t"])

    def estimate(self):
        """-> (d_hat (6,), info).  Zero until the window is full."""
        if not self.buf.ready():
            self.timing.add("rdp", 0.0)
            return np.zeros(6), dict(ready=False, fault=False, ms=0.0)
        y, ms, fault = self.wd(self.rdp.predict, self.buf.window())
        y = np.asarray(y).reshape(-1)
        if not fault:
            self._hist = (self._hist + [y])[-self.smooth_n:]
            y = self.filt(np.mean(self._hist, axis=0))
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
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from geometry_msgs.msg import WrenchStamped
        from std_msgs.msg import Float64MultiArray
        from px4_msgs.msg import ActuatorMotors
    except Exception as ex:                               # noqa: BLE001
        print(f"rdp_estimator: no ROS 2 / px4_msgs environment ({type(ex).__name__}). "
              f"The core is importable and testable without one; see "
              f"EstimatorCore.", file=sys.stderr)
        return 1
    from acmpc_controller import status_msg

    class RDPNode(Node):
        def __init__(self):
            super().__init__("rdp_estimator")
            for k, v in (("model_path", "models/rdp_gru.npz"), ("filter_alpha", 0.0),
                         ("budget_ms", 8.0), ("smooth_n", 32), ("H", 64),
                         ("require_actuator_motors", True)):
                self.declare_parameter(k, v)
            g = lambda k: self.get_parameter(k).value
            self.core = EstimatorCore(g("model_path"), H=int(g("H")),
                                      filt=float(g("filter_alpha")),
                                      budget_ms=float(g("budget_ms")),
                                      smooth_n=int(g("smooth_n")))
            self.require_pwm = bool(g("require_actuator_motors"))
            self.pwm = None
            qos_px4 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(ActuatorMotors, "/fmu/out/actuator_motors",
                                     self.on_motors, qos_px4)
            self.create_subscription(Float64MultiArray, "/acmpc/status",
                                     self.on_status, 10)
            self.pub = self.create_publisher(WrenchStamped,
                                             "/rdp/disturbance_estimate", 10)
            self.pub_st = self.create_publisher(Float64MultiArray, "/rdp/status", 10)
            self.get_logger().info(
                f"RDP up: {self.core.rdp.kind}, H={self.core.H}, smooth_n="
                f"{self.core.smooth_n}, channels={self.core.rdp.channels}")

        def on_motors(self, m):
            self.pwm = np.asarray(m.control[0:4], float)

        def on_status(self, m):
            try:
                self._on_status(m)
            except Exception:                             # noqa: BLE001
                if rclpy.ok():                            # a real error: surface it
                    raise                                 # (else Ctrl-C mid-callback)

        def _on_status(self, m):
            st = status_msg.unpack(m.data)
            if self.pwm is None or not np.all(np.isfinite(self.pwm)):
                if self.require_pwm:
                    self.get_logger().warn(
                        "no /fmu/out/actuator_motors -- the moment channels are "
                        "unobservable, not pushing frames (see README 5.1)",
                        throttle_duration_sec=5.0)
                    return
                pwm = np.zeros(4)
            else:
                pwm = self.pwm
            self.core.push_status(st, pwm)
            d, info = self.core.estimate()
            w = WrenchStamped()
            w.header.stamp = self.get_clock().now().to_msg()
            w.header.frame_id = "map"            # force world ENU, torque body FLU
            w.wrench.force.x, w.wrench.force.y, w.wrench.force.z = (float(z) for z in d[:3])
            w.wrench.torque.x, w.wrench.torque.y, w.wrench.torque.z = (float(z) for z in d[3:])
            self.pub.publish(w)
            s = Float64MultiArray()
            s.data = [float(info["ms"]), float(info["fault"]), float(info["ready"])]
            self.pub_st.publish(s)
            if info["fault"]:
                self.get_logger().warn(
                    f"RDP inference fault -> falling back to zero "
                    f"({self.core.wd.stats()['last_fault']})")

    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=args)
    node = RDPNode()
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
