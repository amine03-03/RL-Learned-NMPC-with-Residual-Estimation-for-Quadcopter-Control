"""Disturbance manager node (§9.7).

Evaluates scenario S0..S6 at 50 Hz, **applies** it to the Gazebo model and
publishes the ground truth.

Subscribes  /fmu/out/vehicle_odometry      attitude, to rotate body torques
            /fmu/out/actuator_motors       rotor commands, for S5/S6
Publishes   /disturbance/ground_truth      geometry_msgs/WrenchStamped
                                           [N world ENU, N.m body FLU]
            /disturbance/phase             std_msgs/String  nominal|disturbed|recovery
            /world/<gz_world>/wrench/persistent  ros_gz_interfaces/EntityWrench
            /world/<gz_world>/wrench/clear       ros_gz_interfaces/Entity

The two ``/world/...`` topics reach gz-sim's ``ApplyLinkWrench`` system (loaded
by PX4's server.config) through ``ros_gz_bridge``; see README §5.8.  The wrench
is applied to the model's canonical link at its origin, force and torque in the
world frame.  ``apply_in_gazebo:=false`` publishes the ground truth only.

Ground truth is for the evaluator and the logger only; §9.1 forbids it reaching
the online controller, and the RDP's input is built from ``/acmpc/status`` and
``actuator_motors`` exclusively, so there is no path for it to.
"""
from __future__ import annotations

import sys

import numpy as np

from .scenarios import PersistentWrench, Scenario, to_world


def phase_of(t, t_on, t_off):
    if t < t_on:
        return "nominal"
    return "disturbed" if t < t_off else "recovery"


def main(args=None):                                       # pragma: no cover
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from geometry_msgs.msg import WrenchStamped
        from std_msgs.msg import String
        from px4_msgs.msg import ActuatorMotors, VehicleOdometry
        from ros_gz_interfaces.msg import Entity, EntityWrench
    except Exception as ex:                               # noqa: BLE001
        print(f"disturbance_manager: no ROS 2 / px4_msgs / ros_gz_interfaces "
              f"environment ({type(ex).__name__}: {ex}).  Scenario definitions are "
              f"importable without one; see scenarios.py.", file=sys.stderr)
        return 1
    from acmpc_controller import frames as F

    class ManagerNode(Node):
        def __init__(self):
            super().__init__("disturbance_manager")
            for k, v in (("scenario", "S0"), ("seed", 0), ("rate_hz", 50.0),
                         ("t_on", 5.0), ("t_off", 12.0), ("apply_in_gazebo", True),
                         ("gz_world", "default"), ("gz_model", "x500_0"),
                         ("settle_s", 1.0)):
                self.declare_parameter(k, v)
            g = lambda k: self.get_parameter(k).value
            self.scen = Scenario(g("scenario"), seed=int(g("seed")),
                                 t_on=float(g("t_on")), t_off=float(g("t_off")))
            self.get_logger().info(f"scenario {self.scen.describe()}")
            qos_px4 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
            self.R = np.eye(3)
            self.pwm = None
            self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry",
                                     self.on_odom, qos_px4)
            self.create_subscription(ActuatorMotors, "/fmu/out/actuator_motors",
                                     self.on_motors, qos_px4)
            self.pub = self.create_publisher(WrenchStamped,
                                             "/disturbance/ground_truth", 10)
            self.pub_phase = self.create_publisher(String, "/disturbance/phase", 10)
            self.apply = bool(g("apply_in_gazebo"))
            w = g("gz_world")
            self.entity = Entity(name=str(g("gz_model")), type=Entity.MODEL)
            self.pub_gz = self.create_publisher(
                EntityWrench, f"/world/{w}/wrench/persistent", 10)
            self.pub_clear = self.create_publisher(Entity, f"/world/{w}/wrench/clear", 10)
            self.pw = PersistentWrench()
            self.settle_s = float(g("settle_s"))
            self.matched_at = None
            if self.apply:
                self.get_logger().info(
                    f"applying to gz model '{self.entity.name}' via "
                    f"/world/{w}/wrench/persistent; waiting for ros_gz_bridge")
            if self.scen.sid in ("S5", "S6"):
                self.get_logger().info("S5/S6 need /fmu/out/actuator_motors")
            self.t0 = None
            self.create_timer(1.0 / float(g("rate_hz")), self.tick)

        def on_odom(self, m):
            _, _, q, _ = F.odometry_to_enu_flu(m.position, m.q, m.velocity,
                                               m.angular_velocity, m.velocity_frame)
            self.R = F.qrotmat(q)

        def on_motors(self, m):
            self.pwm = np.asarray(m.control[0:4], float)

        def _bridge_ready(self, now):
            """The episode clock starts only once the bridge subscribes (plus a
            settle time): an increment published before the match is dropped
            by DDS and would leave a permanent offset in the applied wrench."""
            if not self.apply:
                return True
            if self.matched_at is None:
                if self.pub_gz.get_subscription_count() == 0:
                    self.get_logger().info("waiting for ros_gz_bridge on "
                                           f"{self.pub_gz.topic_name}",
                                           throttle_duration_sec=5.0)
                    return False
                self.matched_at = now
                self.pub_clear.publish(self.entity)          # start from a clean list
            return now - self.matched_at >= self.settle_s

        def tick(self):
            try:
                self._tick()
            except Exception:                             # noqa: BLE001
                if rclpy.ok():                            # a real error: surface it
                    raise                                 # (else Ctrl-C mid-callback)

        def _tick(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.t0 is None:
                if not self._bridge_ready(now):
                    return
                self.t0 = now
                self.get_logger().info("episode clock started")
            t = now - self.t0
            Fw, tau_b = self.scen.wrench(t, R=self.R, pwm=self.pwm)
            m = WrenchStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "map"          # force world ENU, torque body FLU
            m.wrench.force.x, m.wrench.force.y, m.wrench.force.z = (float(z) for z in Fw)
            m.wrench.torque.x, m.wrench.torque.y, m.wrench.torque.z = (float(z) for z in tau_b)
            self.pub.publish(m)
            self.pub_phase.publish(String(data=phase_of(t, self.scen.t_on,
                                                        self.scen.t_off)))
            if not self.apply:
                return
            act, d = self.pw.step(*to_world(Fw, tau_b, self.R))
            if act == "clear":
                self.pub_clear.publish(self.entity)
            elif act == "add":
                ew = EntityWrench()
                ew.header.stamp = m.header.stamp
                ew.entity = self.entity
                ew.wrench.force.x, ew.wrench.force.y, ew.wrench.force.z = (float(z) for z in d[:3])
                ew.wrench.torque.x, ew.wrench.torque.y, ew.wrench.torque.z = (float(z) for z in d[3:])
                self.pub_gz.publish(ew)

        def stop(self):
            if self.apply:                                # leave the world clean
                try:
                    self.pub_clear.publish(self.entity)
                except Exception:                         # noqa: BLE001  context gone
                    pass

    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=args)
    node = ManagerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:                                     # noqa: BLE001
        if rclpy.ok():                                    # not a shutdown race
            raise
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
