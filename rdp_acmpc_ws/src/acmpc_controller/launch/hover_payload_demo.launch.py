"""One leg of the S7 payload demonstration: one controller, hovering, plotted.

    ros2 launch acmpc_controller hover_payload_demo.launch.py controller:=nmpc1
    ros2 launch acmpc_controller hover_payload_demo.launch.py controller:=acmpc_adaptive

The three legs share a session directory, so ``live_zx`` reloads the finished
controllers' traces and the window shows the comparison rather than one curve.
``tools/run_hover_payload_demo.sh`` flies all three back to back.

The estimator is started only for ``acmpc_adaptive``.  Running it for the other
two and simply not subscribing would waste 8 ms of every 20 ms budget on a
prediction nobody reads, and would make the three legs differ in CPU load as
well as in control law -- a confound in the one measurement that is about
timing.
"""
from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def num(name, kind=float):
    """A launch argument as a TYPED node parameter.

    A bare LaunchConfiguration in a parameter dict reaches the node as a
    string, and ROS 2 then guesses the type from how the string looks.  The
    guess is usually right and occasionally is not -- "10" for a parameter the
    node declared as a double arrives as an integer and the node refuses it at
    start-up, which reads as a node that died for no reason.  Say the type.
    """
    return ParameterValue(LaunchConfiguration(name), value_type=kind)


def vec3(a, b, c):
    """Three launch arguments as one double[3] node parameter.

    A Python list of LaunchConfigurations does NOT become a double array: each
    element is a substitution resolving to a string, so the node is handed a
    string array where it declared doubles and rejects it.  Build the literal
    and declare its type instead.
    """
    return ParameterValue(
        PythonExpression(["[float(", a, "), float(", b, "), float(", c, ")]"]),
        value_type=List[float])

#: Only this controller is fed the residual estimate (see controller_node.CONTROLLERS)
ADAPTIVE = "acmpc_adaptive"


def generate_launch_description():
    args = [
        DeclareLaunchArgument("controller", default_value="acmpc",
                              choices=["nmpc1", "acmpc", ADAPTIVE, "pid"]),
        DeclareLaunchArgument("session_dir", default_value="runs/demo_s7"),
        DeclareLaunchArgument("horizon", default_value="10"),
        DeclareLaunchArgument("n_iter", default_value="10"),
        DeclareLaunchArgument("rate_hz", default_value="50.0"),
        DeclareLaunchArgument("hold_x", default_value="0.0"),
        DeclareLaunchArgument("hold_y", default_value="0.0"),
        DeclareLaunchArgument("hold_z", default_value="1.5"),
        DeclareLaunchArgument("payload_mass", default_value="0.30"),
        DeclareLaunchArgument("payload_rx", default_value="0.12"),
        DeclareLaunchArgument("payload_ry", default_value="0.06"),
        DeclareLaunchArgument("payload_rz", default_value="-0.04"),
        DeclareLaunchArgument("model_path", default_value="models/rdp_gru.npz"),
        DeclareLaunchArgument("px4_namespace", default_value=""),
        DeclareLaunchArgument("plot", default_value="true"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("auto_arm", default_value="true"),
    ]
    c = LaunchConfiguration("controller")
    ns = LaunchConfiguration("px4_namespace")
    hold = vec3(LaunchConfiguration("hold_x"), LaunchConfiguration("hold_y"),
                LaunchConfiguration("hold_z"))
    off = vec3(LaunchConfiguration("payload_rx"), LaunchConfiguration("payload_ry"),
               LaunchConfiguration("payload_rz"))
    ns_p = ParameterValue(ns, value_type=str)
    is_adaptive = IfCondition(PythonExpression(["'", c, "' == '", ADAPTIVE, "'"]))

    controller = Node(
        package="acmpc_controller", executable="controller_node",
        name="acmpc_controller", output="screen", emulate_tty=True,
        parameters=[{"controller": ParameterValue(c, value_type=str),
                     "reference": "hold",
                     "p_hold": hold,
                     "horizon": num("horizon", int),
                     "n_iter": num("n_iter", int),
                     "rate_hz": num("rate_hz", float),
                     "auto_arm": num("auto_arm", bool),
                     "px4_namespace": ns_p}])

    estimator = Node(
        package="rdp_estimator", executable="estimator_node",
        name="rdp_estimator", output="screen", emulate_tty=True,
        condition=is_adaptive,
        parameters=[{"model_path": num("model_path", str),
                     "rate_hz": num("rate_hz", float),
                     "px4_namespace": ns_p}])

    truth = Node(
        package="disturbance_manager", executable="manager_node",
        name="disturbance_manager", output="screen", emulate_tty=True,
        parameters=[{"scenario": "S7",
                     "payload_mass": num("payload_mass", float),
                     "payload_offset": off,
                     "px4_namespace": ns_p}])

    plot = Node(
        package="visualization", executable="live_zx",
        name="live_zx", output="screen", emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("plot")),
        parameters=[{"session_dir": num("session_dir", str),
                     "controller": ParameterValue(c, value_type=str),
                     "setpoint": hold,
                     "payload_mass": num("payload_mass", float),
                     "payload_offset": off,
                     "headless": num("headless", bool),
                     "px4_namespace": ns_p}])

    return LaunchDescription(args + [
        LogInfo(msg=["S7 hover / asymmetric payload -- controller ", c,
                     ".  Run `ros2 run acmpc_controller check_px4` first if "
                     "anything looks silent, and confirm the Gazebo payload "
                     "with `python3 tools/payload_sdf.py --status`."]),
        controller, estimator, truth, plot])
