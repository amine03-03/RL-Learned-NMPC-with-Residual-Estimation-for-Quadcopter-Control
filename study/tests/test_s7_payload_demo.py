"""The S7 payload demonstration: topic versioning, the scenario, the estimates.

Everything here is pure Python and runs with no ROS graph, which is the point:
the parts of the demo that can be wrong *silently* -- a topic resolved to the
wrong generation, a mass derived with the wrong sign, a constant duplicated and
then drifted -- are exactly the parts that can be tested without a simulator.
What needs Gazebo is only whether the vehicle flies.
"""
import os

import numpy as np
import pytest

from acmpc_controller import px4_topics as PT
import payload_sdf as PSDF
from acmpc_controller import controller_node as CN
from disturbance_manager import scenarios as SC
from visualization import estimates as EST
from visualization import live_zx as V


# --------------------------------------------------------------------------- #
#  P1..P8  PX4 message-version resolution
# --------------------------------------------------------------------------- #
class _Msg:
    """Stand-in for a generated message class; identity is all select() needs."""


def _importer(known):
    return lambda ts: _Msg if ts in known else None


ODOM = "/fmu/out/vehicle_odometry"


def test_P1_split_version():
    assert PT.split_version("/fmu/out/vehicle_odometry") == (ODOM, 0)
    assert PT.split_version("/fmu/out/vehicle_odometry_v1") == (ODOM, 1)
    assert PT.split_version("/fmu/out/vehicle_odometry_v12") == (ODOM, 12)
    # a trailing _v with no digits is part of the name, not a version
    assert PT.split_version("/a/b_v") == ("/a/b_v", 0)


def test_P2_unversioned_graph_resolves():
    g = [(ODOM, ["px4_msgs/msg/VehicleOdometry"])]
    s = PT.select(ODOM, g, importer=_importer({"px4_msgs/msg/VehicleOdometry"}))
    assert s.topic == ODOM and s.version == 0 and s.msg_class is _Msg


def test_P3_versioned_graph_resolves_to_the_versioned_name():
    """The bug this module exists for: a v1 graph must NOT resolve to the
    unversioned name, because subscribing that name is silently never called."""
    g = [(ODOM + "_v1", ["px4_msgs/msg/VehicleOdometry"])]
    s = PT.select(ODOM, g, importer=_importer({"px4_msgs/msg/VehicleOdometry"}))
    assert s.topic == ODOM + "_v1" and s.version == 1


def test_P4_highest_version_wins_during_a_transition():
    g = [(ODOM, ["px4_msgs/msg/VehicleOdometry"]),
         (ODOM + "_v1", ["px4_msgs/msg/VehicleOdometry"]),
         (ODOM + "_v2", ["px4_msgs/msg/VehicleOdometry"])]
    s = PT.select(ODOM, g, importer=_importer({"px4_msgs/msg/VehicleOdometry"}))
    assert s.version == 2


def test_P5_unimportable_versions_are_skipped_not_preferred():
    """A v4 we cannot deserialise must lose to a v1 we can.  Preferring the
    newest *name* over the newest *usable* type would hand the node a None."""
    g = [(ODOM + "_v1", ["px4_msgs/msg/VehicleOdometry"]),
         (ODOM + "_v4", ["px4_msgs_future/msg/VehicleOdometry"])]
    s = PT.select(ODOM, g, importer=_importer({"px4_msgs/msg/VehicleOdometry"}))
    assert s.version == 1 and s.type_str == "px4_msgs/msg/VehicleOdometry"


def test_P6_all_unimportable_is_a_loud_mismatch():
    g = [(ODOM + "_v4", ["px4_msgs_future/msg/VehicleOdometry"])]
    with pytest.raises(PT.VersionMismatch) as ex:
        PT.select(ODOM, g, importer=_importer(set()))
    m = str(ex.value)
    assert "_v4" in m and "px4_msgs_future" in m and "colcon build" in m


def test_P7_absent_topic_is_None_not_an_exception():
    """An input topic the bridge has not created yet is a wait, not a fault."""
    assert PT.select(ODOM, [("/rosout", ["rcl_interfaces/msg/Log"])],
                     importer=_importer(set())) is None


def test_P8_namespace_and_no_prefix_confusion():
    g = [("/px4_1" + ODOM + "_v1", ["px4_msgs/msg/VehicleOdometry"]),
         (ODOM + "_v1", ["px4_msgs/msg/VehicleOdometry"])]
    imp = _importer({"px4_msgs/msg/VehicleOdometry"})
    assert PT.select(ODOM, g, "/px4_1", imp).topic == "/px4_1" + ODOM + "_v1"
    assert PT.select(ODOM, g, "", imp).topic == ODOM + "_v1"
    # a similarly-named topic must not be captured
    g2 = [(ODOM + "_extra_v1", ["px4_msgs/msg/VehicleOdometry"])]
    assert PT.select(ODOM, g2, "", imp) is None


# --------------------------------------------------------------------------- #
#  S1..S5  the S7 scenario
# --------------------------------------------------------------------------- #
def test_S1_payload_wrench_matches_the_closed_form():
    s = SC.Scenario("S7")
    m_p = s.params["payload_mass"]
    r = np.asarray(s.params["payload_offset"])
    F, tau = s.wrench(0.0)
    assert F == pytest.approx([0.0, 0.0, -m_p * SC.G], abs=1e-12)
    assert tau == pytest.approx([-m_p * SC.G * r[1], m_p * SC.G * r[0], 0.0],
                                abs=1e-12)


def test_S2_payload_is_standing_not_switched():
    """The comparison is about a STANDING error, so the payload is on for the
    whole episode -- including before t_on, where every other scenario is idle."""
    s = SC.Scenario("S7", t_on=5.0, t_off=12.0)
    for t in (0.0, 4.9, 5.0, 11.9, 12.0, 100.0):
        assert np.linalg.norm(s.wrench(t)[0]) > 1.0

def test_S3_attitude_rotates_the_moment_not_the_force():
    """Weight is fixed in the WORLD; the moment it makes is taken in the BODY.

    So a pitch must leave F untouched and change tau -- and it changes tau's
    magnitude too, because the angle between r_p and R^T F is what the cross
    product sees.  (Measured: |tau| 0.3947 -> 0.3408 -> 0.2426 N m at 0, 20 and
    45 deg.)  The invariant that does hold is the Lagrange bound.
    """
    s = SC.Scenario("S7")
    r = np.asarray(s.params["payload_offset"])
    F0, t0 = s.wrench(0.0)
    for deg in (20.0, 45.0, -30.0):
        a = np.radians(deg)
        q = [np.cos(a / 2), 0.0, np.sin(a / 2), 0.0]     # pitch
        F1, t1 = s.wrench(0.0, q)
        assert F1 == pytest.approx(F0, abs=1e-12)        # world force: unchanged
        assert np.linalg.norm(t1 - t0) > 1e-3            # body moment: changed
        assert np.linalg.norm(t1) <= np.linalg.norm(r) * np.linalg.norm(F0) + 1e-12
    # a pitch tilts the payload's lever out of the xy plane, so tau_z appears
    a = np.radians(20.0)
    assert abs(s.wrench(0.0, [np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])[1][2]) > 1e-3


def test_S4_parallel_axis_inertia_is_positive_semidefinite():
    dJ = SC.Scenario("S7").payload_inertia_delta()
    assert np.allclose(dJ, dJ.T)
    assert (np.linalg.eigvalsh(dJ) >= -1e-15).all()


def test_S5_payload_leaves_the_controllers_room_to_differ():
    """A payload that saturates the allocator compares three saturated
    controllers.  Assert the standing pitch moment is a clear minority of the
    authority available WITHOUT dropping out of hover."""
    K_T, OM_MIN, OM_MAX, ARM = 8.54858e-06, 150.0, 1000.0, 0.174
    f_hover = SC.M_NOM * SC.G / 4.0
    head = min(K_T * OM_MAX ** 2 - f_hover, f_hover - K_T * OM_MIN ** 2)
    tau_max = 2 * ARM * head
    _, tau = SC.Scenario("S7").wrench(0.0)
    assert 0.05 < abs(tau[1]) / tau_max < 0.50


# --------------------------------------------------------------------------- #
#  E1..E5  mass and moment estimation
# --------------------------------------------------------------------------- #
def test_E1_mass_recovers_the_payload_from_the_hover_residual():
    tr = EST.payload_truth(0.30, [0.12, 0.06, -0.04])
    m, m_p = EST.mass_estimate(np.array([0.0, 0.0, tr["F_z"]]))
    assert m_p == pytest.approx(0.30, abs=1e-12)
    assert m == pytest.approx(tr["m_total"], abs=1e-12)


def test_E2_mass_sign_a_payload_makes_the_vehicle_HEAVIER():
    """The sign that is easy to get backwards and impossible to see on a plot
    that is already near the truth line."""
    m, m_p = EST.mass_estimate(np.array([0.0, 0.0, -1.0]))
    assert m_p > 0.0 and m > EST.M_NOM


def test_E3_tilt_alone_does_not_bias_the_hover_form():
    """Worth pinning down, because the intuition is wrong.

    In a steady tilted hold ``T cos(tilt) = m g`` exactly, so
    ``F_z = -T cos(tilt) m_p/m = -m_p g`` and the hover form (9.10) is EXACT at
    any tilt.  Tilt is not what breaks it.
    """
    m_p_true, a = 0.30, np.radians(25.0)
    q = np.array([np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])
    m_true = EST.M_NOM + m_p_true
    T = m_true * EST.G / np.cos(a)                       # hold altitude at tilt
    zb = EST.qrotmat(q)[:, 2]
    F = -T * zb * (m_p_true / m_true)                    # (9.9), exactly
    assert EST.mass_estimate(F, T, q)[1] == pytest.approx(m_p_true, rel=1e-12)
    assert EST.mass_estimate(F)[1] == pytest.approx(m_p_true, rel=1e-12)


def test_E3b_vertical_acceleration_is_what_biases_the_hover_form():
    """``T cos(tilt) = m (g + a_z)``, so (9.10) reads ``m_p (g + a_z) / g``:
    +10.2 % per 1 m/s^2 of climb.  The exact branch is unaffected.  Measured
    m_p_hat at a_z = +2 m/s^2: hover 0.36118, exact 0.30000."""
    m_p_true, a = 0.30, np.radians(25.0)
    q = np.array([np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])
    m_true = EST.M_NOM + m_p_true
    zb = EST.qrotmat(q)[:, 2]
    for a_z in (1.0, 2.0, -1.5):
        T = m_true * (EST.G + a_z) / np.cos(a)
        F = -T * zb * (m_p_true / m_true)
        exact = EST.mass_estimate(F, T, q)[1]
        hover = EST.mass_estimate(F)[1]
        assert exact == pytest.approx(m_p_true, rel=1e-12)
        assert hover == pytest.approx(m_p_true * (EST.G + a_z) / EST.G, rel=1e-9)
        assert abs(exact - m_p_true) < abs(hover - m_p_true)


def test_E4_zero_thrust_frames_fall_back_instead_of_exploding():
    """A logged run has a few frames before the motors spin up.  Dividing by
    that thrust gives an infinity that takes the whole plot's y range with it."""
    m_p_true = 0.30
    F = np.tile(np.array([0.0, 0.0, -m_p_true * EST.G]), (4, 1))
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1))
    T = np.array([(EST.M_NOM + m_p_true) * EST.G, 0.0,
                  (EST.M_NOM + m_p_true) * EST.G, 0.0])   # two dead frames
    m, m_p = EST.mass_estimate(F, T, q)
    assert np.isfinite(m).all() and np.isfinite(m_p).all()
    assert m_p == pytest.approx(m_p_true, abs=1e-9)       # both branches agree


def test_E5_cg_offset_recovers_r_and_refuses_when_unidentifiable():
    tr = EST.payload_truth(0.30, [0.12, 0.06, -0.04])
    rx, ry = EST.cg_offset_estimate(tr["tau"], tr["m_p"])
    assert (float(rx), float(ry)) == pytest.approx((0.12, 0.06), abs=1e-12)
    # a payload below the identifiability floor must give NaN, not a huge number
    rx2, ry2 = EST.cg_offset_estimate(np.array([1e-4, 1e-4, 0.0]), 1e-4)
    assert np.isnan(rx2) and np.isnan(ry2)


# --------------------------------------------------------------------------- #
#  R1..R4  the recorder and the plot data
# --------------------------------------------------------------------------- #
def _fill(rec, name, n=100, bias=(0.1, -0.05)):
    rec.select(name)
    for k in range(n):
        rec.push_pose(k * 0.02, (bias[0], 0.0, 1.5 + bias[1]))
    return rec


def test_R1_traces_round_trip_through_the_npz(tmp_path):
    rec = V.DemoRecorder(setpoint=(0, 0, 1.5))
    for n, b in zip(V.CONTROLLERS, [(0.3, -0.2), (0.17, -0.1), (0.03, -0.01)]):
        _fill(rec, n, bias=b)
    p = rec.save(str(tmp_path / "zx_traces.npz"))
    back = V.DemoRecorder(setpoint=(0, 0, 1.5)).load(p)
    assert set(back.metrics()) == set(rec.metrics())
    for n in V.CONTROLLERS:
        assert back.metrics()[n]["z_bias"] == pytest.approx(
            rec.metrics()[n]["z_bias"], abs=1e-12)


def test_R2_loading_does_not_clobber_the_controller_in_flight(tmp_path):
    """Three runs share one file; reloading it mid-flight must not discard the
    samples the live controller has already taken."""
    a = _fill(V.DemoRecorder(), "nmpc1", n=100)
    p = a.save(str(tmp_path / "t.npz"))
    b = V.DemoRecorder()
    _fill(b, "nmpc1", n=7)                   # same name, now flying
    b.load(p)
    assert len(b.traces["nmpc1"]) == 7


def test_R3_estimate_channels_stay_aligned_with_the_pose_column():
    """The estimator is slower than the pose stream; every column must still be
    the same length or the arrays cannot be stacked for the plot."""
    rec = V.DemoRecorder()
    rec.select("acmpc_adaptive")
    for k in range(50):
        rec.push_pose(k * 0.02, (0.0, 0.0, 1.5))
        if k % 3 == 0:
            rec.set_estimate(np.arange(6) * 0.1)
    a = rec.traces["acmpc_adaptive"].arrays()
    assert len({v.size for v in a.values()}) == 1


def test_R4_metrics_measure_the_standing_offset_not_the_transient():
    """A run that starts at the setpoint and drifts to a bias must report the
    bias.  An RMS from t=0 would report roughly half of it."""
    rec = V.DemoRecorder(setpoint=(0, 0, 1.5))
    rec.select("nmpc1")
    for k in range(1000):                    # 20 s: 10 s ramp, 10 s settled
        t = k * 0.02
        s = min(t / 10.0, 1.0)
        rec.push_pose(t, (0.3 * s, 0.0, 1.5 - 0.2 * s))
    m = rec.metrics()["nmpc1"]
    assert m["x_bias"] == pytest.approx(0.30, abs=0.01)
    assert m["z_bias"] == pytest.approx(-0.20, abs=0.01)


def test_R5_thrust_from_actuator_motors_respects_the_idle_floor():
    """PX4 maps c onto [OM_MIN, OM_MAX] (A2); c = 0 is 0.77 N, not zero."""
    assert V.thrust_from_actuator_motors([0, 0, 0, 0]) == pytest.approx(
        4 * 8.54858e-06 * 150.0 ** 2, rel=1e-12)
    assert V.thrust_from_actuator_motors([1, 1, 1, 1]) == pytest.approx(
        4 * 8.54858e-06 * 1000.0 ** 2, rel=1e-12)
    # and the hover collective must give the vehicle's weight back
    m_g = SC.M_NOM * SC.G
    assert V.thrust_from_actuator_motors([0.728741895] * 4) == pytest.approx(
        m_g, rel=2e-3)


# --------------------------------------------------------------------------- #
#  C1..C4  no constant is duplicated without being asserted equal
# --------------------------------------------------------------------------- #
def test_C1_payload_defaults_agree_between_the_sdf_patch_and_the_scenario():
    """The SDF sets the plant; the scenario sets the truth line on the plot.
    If they drift, every plotted truth is wrong and nothing raises."""
    assert PSDF.DEFAULT_MASS == SC.S7_PAYLOAD_M
    assert tuple(PSDF.DEFAULT_OFFSET) == tuple(SC.S7_PAYLOAD_R)


def test_C2_mass_and_gravity_agree_across_the_workspace():
    import x500_core_jax as X
    for mod in (SC, EST):
        assert mod.M_NOM == pytest.approx(float(X.M_TOT), abs=1e-15)
        assert mod.G == pytest.approx(float(X.P.g), abs=1e-15)


def test_C3_rotor_constants_in_the_plotter_match_the_study():
    import x500_core_jax as X
    assert V.K_T == X.P.K_T
    assert V.OM_MIN == X.P.Om_min and V.OM_MAX == X.P.Om_max


def test_C4_controller_table_routes_the_estimate_to_exactly_one_controller():
    """'adaptive' must be the ONLY one that sees d_hat, or the comparison is
    between three things with the same information."""
    assert [n for n, s in CN.CONTROLLERS.items() if s["use_d"]] == \
        ["acmpc_adaptive"]
    assert CN.CONTROLLERS["nmpc1"]["mode"] == "nmpc1"
    assert CN.CONTROLLERS["acmpc"]["mode"] == "acmpc"
    assert CN.CONTROLLERS["acmpc_adaptive"]["mode"] == "acmpc"
    assert set(V.CONTROLLERS) <= set(CN.CONTROLLERS)


# --------------------------------------------------------------------------- #
#  D1..D3  the SDF patch
# --------------------------------------------------------------------------- #
_SDF = """<?xml version="1.0"?>
<sdf version='1.9'><model name='x500'>
  <link name="base_link"><inertial><mass>2.0</mass></inertial></link>
</model></sdf>
"""


def test_D1_install_is_idempotent_and_revert_is_exact(tmp_path):
    p = tmp_path / "model.sdf"
    p.write_text(_SDF)
    PSDF.install(str(p), 0.3, (0.12, 0.06, -0.04))
    PSDF.install(str(p), 0.3, (0.12, 0.06, -0.04))
    assert p.read_text().count("acmpc_payload_joint") == 1
    PSDF.revert(str(p))
    assert p.read_text() == _SDF


def test_D2_patched_sdf_is_well_formed_xml_with_the_right_pose(tmp_path):
    import xml.etree.ElementTree as ET
    p = tmp_path / "model.sdf"
    p.write_text(_SDF)
    PSDF.install(str(p), 0.3, (0.12, 0.06, -0.04))
    root = ET.parse(str(p)).getroot()
    link = root.find(".//link[@name='acmpc_payload']")
    assert link is not None
    assert float(link.find("inertial/mass").text) == pytest.approx(0.3)
    assert [float(v) for v in link.find("pose").text.split()][:3] == \
        pytest.approx([0.12, 0.06, -0.04])
    j = root.find(".//joint[@name='acmpc_payload_joint']")
    assert j.get("type") == "fixed" and j.find("parent").text == "base_link"


def test_D3_a_wrong_parent_link_is_refused_with_the_options(tmp_path):
    p = tmp_path / "model.sdf"
    p.write_text(_SDF)
    with pytest.raises(SystemExit) as ex:
        PSDF.install(str(p), 0.3, (0.1, 0, 0), parent="not_a_link")
    assert "base_link" in str(ex.value)
    assert p.read_text() == _SDF                 # and nothing was written


def test_R6_exact_mass_branch_is_used_when_thrust_and_attitude_are_recorded():
    """The panel labels the mode, so the mode must be real.  During a 2 m/s^2
    climb the hover form reads 20.4 % high; the exact inversion does not."""
    rec = V.DemoRecorder()
    rec.select("acmpc_adaptive")
    m_p, ang = 0.30, np.radians(8.0)
    q = np.array([np.cos(ang / 2), 0.0, np.sin(ang / 2), 0.0])
    m_true = EST.M_NOM + m_p
    zb = EST.qrotmat(q)[:, 2]
    for k in range(200):
        T = m_true * (EST.G + 2.0) / np.cos(ang)
        F = -T * zb * (m_p / m_true)
        rec.push_pose(k * 0.02, (0.0, 0.0, 1.5), q)
        rec.set_estimate(np.concatenate([F, [0.0, 0.0, 0.0]]), thrust=T)
    tr = rec.traces["acmpc_adaptive"]
    m_hat, mp_hat, exact = tr.mass()
    assert exact is True
    assert np.mean(mp_hat) == pytest.approx(m_p, rel=1e-9)
    hover = EST.mass_estimate(tr.force())[1]
    assert np.mean(hover) == pytest.approx(m_p * (EST.G + 2.0) / EST.G, rel=1e-9)


def test_R7_no_attitude_falls_back_to_the_hover_form_and_says_so():
    rec = V.DemoRecorder()
    rec.select("acmpc_adaptive")
    for k in range(50):
        rec.push_pose(k * 0.02, (0.0, 0.0, 1.5))      # no q
        rec.set_estimate(np.array([0, 0, -0.30 * EST.G, 0, 0, 0]))
    m_hat, mp_hat, exact = rec.traces["acmpc_adaptive"].mass()
    assert exact is False
    assert np.mean(mp_hat) == pytest.approx(0.30, rel=1e-9)


def test_R8_a_trace_file_without_the_attitude_columns_still_loads(tmp_path):
    """Sessions recorded before the attitude columns existed must not crash the
    plotter -- they simply get the hover form."""
    p = tmp_path / "old.npz"
    n = 40
    np.savez_compressed(p, **{"nmpc1/t": np.arange(n) * 0.02,
                              "nmpc1/x": np.zeros(n), "nmpc1/y": np.zeros(n),
                              "nmpc1/z": np.full(n, 1.5),
                              "__setpoint__": np.array([0.0, 0.0, 1.5])})
    rec = V.DemoRecorder().load(str(p))
    a = rec.traces["nmpc1"].arrays()
    assert len({v.size for v in a.values()}) == 1 and a["t"].size == n
    assert rec.traces["nmpc1"].mass()[2] is False
    assert rec.metrics()["nmpc1"]["n"] == n


def test_P9_a_tie_within_one_version_is_deterministic():
    """Two importable types on one topic (an old and a new message package both
    on the path) must not resolve to whichever the graph listed first."""
    g_a = [(ODOM + "_v1", ["px4_msgs_old/msg/VehicleOdometryV1",
                           "px4_msgs/msg/VehicleOdometry"])]
    g_b = [(ODOM + "_v1", ["px4_msgs/msg/VehicleOdometry",
                           "px4_msgs_old/msg/VehicleOdometryV1"])]
    imp = _importer({"px4_msgs/msg/VehicleOdometry",
                     "px4_msgs_old/msg/VehicleOdometryV1"})
    assert PT.select(ODOM, g_a, importer=imp).type_str == \
        PT.select(ODOM, g_b, importer=imp).type_str == "px4_msgs/msg/VehicleOdometry"


# --------------------------------------------------------------------------- #
#  V1..V3  vector parameters, however ROS 2 delivers them
# --------------------------------------------------------------------------- #
def test_V1_vector_parameter_accepts_every_shape_launch_can_deliver():
    """A launch file can hand a node a double[] as floats, as strings, or as
    one string.  The node dies during construction on the unexpected shape,
    which in the launch output looks like a node that died for no reason."""
    for v in ([0.0, 0.0, 1.5], ["0.0", "0.0", "1.5"], "[0.0, 0.0, 1.5]",
              "0.0 0.0 1.5", (0, 0, 1.5), np.array([0.0, 0.0, 1.5])):
        assert PT.as_floats(v, 3, "p_hold") == pytest.approx([0.0, 0.0, 1.5])


def test_V2_vector_parameter_refuses_the_wrong_length_or_content():
    for bad in ("[1,2]", [1, 2, 3, 4], "abc", ["1", "x", "3"]):
        with pytest.raises(ValueError):
            PT.as_floats(bad, 3, "p_hold")


def test_V3_vector_parameter_names_itself_in_the_error():
    """The message is the whole value of this helper: it has to say which
    parameter was wrong, in a launch log carrying four nodes' output."""
    with pytest.raises(ValueError) as ex:
        PT.as_floats("[1,2]", 3, "payload_offset")
    assert "payload_offset" in str(ex.value)


# --------------------------------------------------------------------------- #
#  O1..O5  the observation the cost map is actually handed
# --------------------------------------------------------------------------- #
import copy
import pickle

import x500_core_jax as X


def _ckpt():
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "..", "artifacts", "acmpc", "model.pkl")
    if not os.path.exists(p):
        pytest.skip("no AC-MPC checkpoint in artifacts/")
    return pickle.load(open(p, "rb"))


def _widen(ck, extra=6):
    """The 53-wide layout nb5's variant C produces: OBS_DIM + the 6 oracle
    channels, with the trunk's first layer widened to match."""
    import jax.numpy as jnp
    a = copy.deepcopy(ck["actor"])
    W, b = a["trunk"][0]
    a["trunk"][0] = (jnp.asarray(np.vstack([np.asarray(W),
                                            np.zeros((extra, W.shape[1]))])), b)
    for k, fill in (("mu", 0.0), ("var", 1.0)):
        v = np.asarray(a["obs_norm"][k])
        a["obs_norm"][k] = jnp.asarray(np.concatenate([v, np.full(extra, fill)]))
    return dict(ck, actor=a, cfg=dict(ck["cfg"], oracle=True))


def _ctrl(**kw):
    from acmpc_controller.controller import ACMPCController
    return ACMPCController(mode="acmpc", horizon=1, n_iter=5, use_d=True, **kw)


def _stage(c, d_hat=None, x=(0.1, 0.0, 1.4)):
    from acmpc_controller import refgen as RG
    xr, ur = RG.ref_traj(lambda t: (np.array([0.0, 0.0, 1.5]), np.zeros(3),
                                    np.zeros(3)), 0.0, 1, 0.02)
    x13 = np.concatenate([x, np.zeros(3), [1, 0, 0, 0], np.zeros(3)])
    prev = np.tile(np.concatenate([np.array([0.0, 0.0, 1.5]) - np.asarray(x),
                                   np.zeros(3)]), (3, 1))
    return c.step(x13, xr, ur, d_hat=d_hat, preview=prev)


def test_O1_observation_is_built_not_zeroed():
    """The cost map is a FUNCTION of the observation.  Handing it zeros
    evaluates the learned cost at a point that never occurs in training, and
    nothing raises -- the widths match."""
    c = _ctrl(ckpt=_ckpt())
    _, info = _stage(c, np.zeros(6))
    assert info["obs"] is not None
    assert info["obs"].size == X.OBS_DIM
    assert np.abs(info["obs"]).max() > 0.0


def test_O2_error_and_preview_blocks_land_where_5_3_puts_them():
    c = _ctrl(ckpt=_ckpt())
    _, info = _stage(c, np.zeros(6), x=(0.1, 0.0, 1.4))
    o = info["obs"]
    assert o[:X.NE] == pytest.approx(info["e"], abs=1e-12)     # e first
    pv = o[X.NE:X.NE + 18].reshape(3, 6)
    for row in pv:                                            # p_ref - p, v_ref
        assert row[0:3] == pytest.approx([-0.1, 0.0, 0.1], abs=1e-12)
        assert row[3:6] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_O3_observation_memory_advances_like_the_training_env():
    """int_ep integrates the position error and the EMAs move; a controller
    that never advances them hands the cost map four blocks of zeros for the
    whole flight."""
    c = _ctrl(ckpt=_ckpt())
    for _ in range(10):
        _stage(c, np.zeros(6))
    assert np.abs(c.int_ep).max() > 0.0
    assert np.abs(c.du_bar).max() > 0.0
    # the integral of a constant 0.1 m error over 10 steps of 20 ms.  The
    # error block is x - x_ref, so holding 0.1 m PAST the setpoint integrates
    # POSITIVE -- while the preview block is p_ref - p and is negative.  The
    # two sign conventions are the study's and sit side by side in (5.3).
    assert c.int_ep[0] == pytest.approx(+0.1 * 10 * X.P.dt_c, rel=1e-9)


def test_O4_a_variant_C_checkpoint_gets_its_six_oracle_channels():
    """The crash this fixes: variant C was trained with the residual in the
    observation too, so its actor expects OBS_DIM + 6.  Handing it OBS_DIM is
    a broadcasting error inside the cost map, a long way from its cause."""
    ck = _widen(_ckpt())
    c = _ctrl(ckpt=ck)
    assert c.obs_w == X.OBS_DIM + 6 and c.oracle_n == 6
    d = np.array([0.0, 0.0, -2.942, -0.1765, 0.3530, 0.0])
    _, info = _stage(c, d)
    assert info["obs"].size == X.OBS_DIM + 6
    assert info["obs"][-6:] == pytest.approx(d, abs=1e-12)     # raw wrench


def test_O5_oracle_target_residual_converts_instead_of_passing_raw():
    """'wrench' and 'residual' have the same width, so choosing wrongly is
    silent.  The training config says which."""
    ck = _widen(_ckpt())
    ck = dict(ck, cfg=dict(ck["cfg"], oracle_target="residual"))
    c = _ctrl(ckpt=ck)
    d = np.array([0.0, 0.0, -2.942, -0.1765, 0.3530, 0.0])
    _, info = _stage(c, d)
    assert info["obs"][-6:] == pytest.approx(
        np.asarray(X.wrench_to_dmod(d[None]))[0], rel=1e-9)


def test_O6_an_unrecognisable_observation_width_is_refused_at_construction():
    import jax.numpy as jnp
    ck = _ckpt()
    a = copy.deepcopy(ck["actor"])
    for k in ("mu", "var"):
        v = np.asarray(a["obs_norm"][k])
        a["obs_norm"][k] = jnp.asarray(np.concatenate([v, np.zeros(3)]))
    with pytest.raises(ValueError) as ex:
        _ctrl(ckpt=dict(ck, actor=a))
    assert "50-wide" in str(ex.value)


def test_O7_study_directory_is_found_by_search_not_by_counting_dots():
    """With `colcon build --symlink-install` the module that runs lives under
    build/, at a different depth from src/.  Counting `..` worked by
    coincidence; the estimator's copy did not exist at all, which is why
    rdp_infer was missing."""
    import os
    d = PT.add_study_to_path()
    assert os.path.isfile(os.path.join(d, "rdp_infer.py"))
    assert os.path.isfile(os.path.join(d, "x500_core_jax.py"))


# --------------------------------------------------------------------------- #
#  H1..H3  the horizon is baked into the checkpoint
# --------------------------------------------------------------------------- #
def test_H1_horizon_comes_from_the_checkpoint_not_the_caller():
    """The cost-map head emits one parameter block per stage from a single
    dense layer, so its output width is N_train * REP_DIM and N is fixed at
    training time.  Asking an N=1 head to fill ten stages fails inside the cost
    map as 'cannot reshape (1, 40) into (1, 10, 40)' -- nowhere near the
    horizon parameter that caused it."""
    from acmpc_controller.controller import ACMPCController
    ck = _ckpt()
    n_train = int(ck["cfg"]["N"])
    c = ACMPCController(mode="acmpc", horizon=n_train + 9, n_iter=5, ckpt=ck)
    assert c.N == n_train
    assert c.cfg["N"] == n_train


def test_H2_the_head_width_is_what_forces_it():
    """Pin the arithmetic, so a future representation change cannot quietly
    make the override look safe again."""
    ck = _ckpt()
    W, _ = ck["actor"]["head"][0]
    assert W.shape[1] == int(ck["cfg"]["N"]) * X.REP_DIM[ck["cfg"]["rep"]]


def test_H3_a_checkpoint_horizon_still_solves_end_to_end():
    from acmpc_controller.controller import ACMPCController
    ck = _ckpt()
    c = ACMPCController(mode="acmpc", horizon=10, n_iter=5, use_d=True, ckpt=ck)
    from acmpc_controller import refgen as RG
    xr, ur = RG.ref_traj(lambda t: (np.array([0.0, 0.0, 1.5]), np.zeros(3),
                                    np.zeros(3)), 0.0, c.N, 0.02)
    assert xr.shape[0] == c.N + 1          # the reference must match the solver
    x13 = np.concatenate([[0.1, 0.0, 1.4], np.zeros(3), [1, 0, 0, 0], np.zeros(3)])
    prev = np.tile(np.concatenate([np.array([0.0, 0.0, 1.5]) - x13[0:3],
                                   np.zeros(3)]), (3, 1))
    u, _ = c.step(x13, xr, ur, d_hat=np.zeros(6), preview=prev)
    assert np.isfinite(u).all()


def test_H4_the_controller_table_fixes_nmpc1_at_one_stage():
    """'nmpc1' is a name with a number in it; the ledger reports N=1 for it."""
    assert CN.CONTROLLERS["nmpc1"]["horizon"] == 1
    # the learned ones defer to their checkpoint
    for n in ("acmpc", "acmpc_adaptive"):
        assert CN.CONTROLLERS[n]["horizon"] is None


# --------------------------------------------------------------------------- #
#  A1  ROS arguments must reach rclpy, not argparse
# --------------------------------------------------------------------------- #
def test_A1_replay_is_detected_with_ros_args_present():
    """live_zx stripped the '--ros-args' marker and handed the remainder to
    rclpy.init, which could then not parse '--params-file' -- so every ROS
    parameter silently fell back to its declared default and the node wrote to
    runs/demo_s7 instead of the session it was launched with.  Nothing raised:
    a default is a legal value.  This pins the split."""
    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
        "rdp_acmpc_ws", "src", "visualization", "visualization",
        "live_zx.py")).read()
    # argparse must be fed remove_ros_args(), and rclpy.init the original argv
    assert "remove_ros_args(args=argv)" in src
    assert "rclpy.init(args=argv)" in src
    assert "if a != \"--ros-args\"" not in src        # the old, wrong filter



# --------------------------------------------------------------------------- #
#  W1..W2  the JIT must be compiled before the control loop, not inside it
# --------------------------------------------------------------------------- #
def test_W1_warmup_makes_the_first_real_tick_fit_the_control_period():
    """JAX compiles on the first call: measured 13.5 s on a CPU-only jaxlib
    against a steady 7.4 ms.  Paying that inside the first timer callback
    blocks the executor, so no OffboardControlMode is published for 13 s --
    and PX4 refuses offboard, or drops out of it, after 0.5 s of silence.  The
    vehicle would never arm and the log would show a healthy controller."""
    import time as _t
    from acmpc_controller.controller import ACMPCController
    from acmpc_controller import refgen as RG
    c = ACMPCController(mode="acmpc", horizon=1, n_iter=10, use_d=True,
                        ckpt=_ckpt())
    c.warmup()
    xr, ur = RG.ref_traj(lambda t: (np.array([0.0, 0.0, 1.5]), np.zeros(3),
                                    np.zeros(3)), 0.0, c.N, 0.02)
    x13 = np.concatenate([[0.05, 0.0, 1.47], np.zeros(3), [1, 0, 0, 0],
                          np.zeros(3)])
    t0 = _t.perf_counter()
    c.step(x13, xr, ur, d_hat=np.zeros(6), preview=np.zeros((3, 6)))
    first = _t.perf_counter() - t0
    assert first < 0.020, (f"first tick after warm-up took {first*1e3:.0f} ms, "
                           f"which does not fit the 20 ms control period")


def test_W2_warmup_leaves_no_state_behind():
    """Warm-up flies a synthetic hover.  If it seeded the rotor observer or
    (5.3)'s memory, the first real tick would start from a fiction."""
    from acmpc_controller.controller import ACMPCController
    c = ACMPCController(mode="acmpc", horizon=1, n_iter=5, use_d=True,
                        ckpt=_ckpt())
    c.warmup()
    assert np.abs(c.int_ep).max() == 0.0
    assert np.abs(c.du_bar).max() == 0.0
    assert np.abs(c.ev_bar).max() == 0.0
    assert np.abs(c.om_bar).max() == 0.0
    assert c.Om_hat == pytest.approx(np.full(4, float(X.OM_HOVER)))
    assert c.solve_ms == []


# --------------------------------------------------------------------------- #
#  M1..M6  the arm handshake must be confirmed, not assumed
# --------------------------------------------------------------------------- #
def test_M1_both_halves_are_required_before_the_run_counts_as_armed():
    """Armed but not OFFBOARD is the dangerous half: PX4 is flying on its own
    controller and discarding every rate setpoint we publish, so the trace is
    flat for a reason that has nothing to do with the controller under test."""
    A, O = CN.ARMING_STATE_ARMED, CN.NAVIGATION_STATE_OFFBOARD
    assert CN.handshake_done(True, A, O) is True
    assert CN.handshake_done(True, A, 4) is False          # armed, AUTO.LOITER
    assert CN.handshake_done(True, 1, O) is False          # offboard, STANDBY
    assert CN.handshake_done(True, 1, 4) is False


def test_M2_no_status_message_yet_is_not_a_confirmation():
    """The old code latched ``armed = True`` the moment it sent the request.
    With the status topic present but no message delivered yet, arming_state is
    None -- which must read as 'not yet', not as 'fine'."""
    assert CN.handshake_done(True, None, None, tries=1) is False
    assert CN.handshake_done(True, None, None, tries=99) is False


def test_M3_a_missing_status_topic_falls_back_to_open_loop():
    """No vehicle_status on the graph at all: there is nothing to read back, so
    one sent request is all the confirmation available.  It must not block."""
    assert CN.handshake_done(False, None, None, tries=0) is False
    assert CN.handshake_done(False, None, None, tries=1) is True


def test_M4_only_the_half_that_is_still_missing_is_re_requested():
    A, O = CN.ARMING_STATE_ARMED, CN.NAVIGATION_STATE_OFFBOARD
    assert CN.handshake_requests(True, 1, 4) == (True, True)     # neither yet
    assert CN.handshake_requests(True, A, 4) == (True, False)    # armed only
    assert CN.handshake_requests(True, 1, O) == (False, True)    # offboard only
    assert CN.handshake_requests(True, A, O) == (False, False)   # done
    # with no status to read, always ask for both
    assert CN.handshake_requests(False, None, None) == (True, True)


def test_M5_the_codes_are_read_off_the_message_when_it_carries_them():
    """px4_topics exists so that nothing about the wire format is hard-coded.
    The arming codes are part of the wire format, so the node prefers the
    constants on the resolved class and only falls back to the literals."""
    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
        "rdp_acmpc_ws", "src", "acmpc_controller", "acmpc_controller",
        "controller_node.py")).read()
    assert 'getattr(cls, "ARMING_STATE_ARMED"' in src
    assert 'getattr(cls, "NAVIGATION_STATE_OFFBOARD"' in src
    # the handshake is never latched on the request alone any more: the only
    # place that sets armed=True is the branch guarded by _handshake_done()
    assert "PX4 confirms ARMED + OFFBOARD" in src
    assert src.count("self.armed = True") == 1
    assert "_handshake_done()" in src


def test_M6_status_is_resolved_like_every_other_px4_topic():
    """A hard-coded '/fmu/out/vehicle_status' would be silently never called on
    the user's graph, which carries it as '_v4'."""
    spec = PT.select("/fmu/out/vehicle_status",
                     [("/fmu/out/vehicle_status_v4",
                       ["px4_msgs/msg/VehicleStatus"])],
                     importer=lambda t: object)
    assert spec.topic == "/fmu/out/vehicle_status_v4" and spec.version == 4
    assert "/fmu/out/vehicle_status" in PT.DEFAULT_BASES


# --------------------------------------------------------------------------- #
#  T1..T4  the actuation path must not restate the study's constants
# --------------------------------------------------------------------------- #
def test_T1_the_rate_box_is_exactly_the_reachable_set():
    """The node publishes ``OM_MAX * u[1:4]`` with no clamp of its own.  That
    is only correct because the solver's input box IS the reachable rate set:
    U_HI[1:] = rate_max / om_max, so the product lands exactly on rate_max.
    Move either constant without the other and the node starts commanding
    rates the control model never planned for -- which on the vehicle reads as
    a controller that is merely poor, not as a mis-wiring."""
    assert np.asarray(X.OM_MAX) * np.asarray(X.U_HI)[1:] == \
        pytest.approx(np.full(3, float(X.P.rate_max)))
    assert np.asarray(X.OM_MAX) * np.asarray(X.U_LO)[1:] == \
        pytest.approx(np.full(3, -float(X.P.rate_max)))


def test_T2_the_collective_box_is_what_px4_thrust_body_accepts():
    """``thrust_body`` is a normalised [-1, 0] on the FRD z axis, so the
    collective must already be bounded to [0, 1] before the sign flip."""
    assert float(X.U_LO[0]) == 0.0 and float(X.U_HI[0]) == 1.0


def test_T3_the_node_reads_the_constants_it_does_not_rewrite_them():
    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..",
        "rdp_acmpc_ws", "src", "acmpc_controller", "acmpc_controller",
        "controller_node.py")).read()
    assert "self.ctrl.X.OM_MAX" in src
    assert "[10.0, 10.0, 4.0]" not in src          # the literal it used to hold
    # and the FRD conversion goes through the tested helper, not an inline -u[0]
    assert "F.ctbr_to_px4_thrust(u[0])" in src
    assert "float(-u[0])" not in src


def test_T4_refgen_om_max_still_matches_the_study():
    """refgen restates the study's constants in pure NumPy on purpose; that is
    only safe while something checks them."""
    from acmpc_controller import refgen as RG
    assert RG.OM_MAX == pytest.approx(np.asarray(X.OM_MAX))
    assert RG.M_NOM == pytest.approx(float(X.M_TOT))
