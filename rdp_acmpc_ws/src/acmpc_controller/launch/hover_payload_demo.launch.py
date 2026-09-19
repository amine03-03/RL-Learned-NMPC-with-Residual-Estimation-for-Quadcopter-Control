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
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

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
    hold = [LaunchConfiguration("hold_x"), LaunchConfiguration("hold_y"),
            LaunchConfiguration("hold_z")]
    off = [LaunchConfiguration("payload_rx"), LaunchConfiguration("payload_ry"),
           LaunchConfiguration("payload_rz")]
    is_adaptive = IfCondition(PythonExpression(["'", c, "' == '", ADAPTIVE, "'"]))

    controller = Node(
        package="acmpc_controller", executable="controller_node",
        name="acmpc_controller", output="screen", emulate_tty=True,
        parameters=[{"controller": c,
                     "reference": "hold",
                     "p_hold": hold,
                     "horizon": LaunchConfiguration("horizon"),
                     "n_iter": LaunchConfiguration("n_iter"),
                     "rate_hz": LaunchConfiguration("rate_hz"),
                     "auto_arm": LaunchConfiguration("auto_arm"),
                     "px4_namespace": ns}])

    estimator = Node(
        package="rdp_estimator", executable="estimator_node",
        name="rdp_estimator", output="screen", emulate_tty=True,
        condition=is_adaptive,
        parameters=[{"model_path": LaunchConfiguration("model_path"),
                     "rate_hz": LaunchConfiguration("rate_hz"),
                     "px4_namespace": ns}])

    truth = Node(
        package="disturbance_manager", executable="manager_node",
        name="disturbance_manager", output="screen", emulate_tty=True,
        parameters=[{"scenario": "S7",
                     "payload_mass": LaunchConfiguration("payload_mass"),
                     "payload_offset": off,
                     "px4_namespace": ns}])

    plot = Node(
        package="visualization", executable="live_zx",
        name="live_zx", output="screen", emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("plot")),
        parameters=[{"session_dir": LaunchConfiguration("session_dir"),
                     "controller": c,
                     "setpoint": hold,
                     "payload_mass": LaunchConfiguration("payload_mass"),
                     "payload_offset": off,
                     "headless": LaunchConfiguration("headless"),
                     "px4_namespace": ns}])

    return LaunchDescription(args + [
        LogInfo(msg=["S7 hover / asymmetric payload -- controller ", c,
                     ".  Run `ros2 run acmpc_controller check_px4` first if "
                     "anything looks silent, and confirm the Gazebo payload "
                     "with `python3 tools/payload_sdf.py --status`."]),
        controller, estimator, truth, plot])
