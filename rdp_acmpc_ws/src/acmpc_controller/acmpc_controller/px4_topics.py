"""px4_topics -- resolve PX4 uORB topics across message-versioning generations.

PX4 v1.16 introduced **message versioning**: a uORB message that changes its
ABI keeps its old definition alive under a version suffix, and the uXRCE-DDS
bridge advertises the versioned name on the wire::

    /fmu/out/vehicle_odometry          <- pre-1.16, unversioned
    /fmu/out/vehicle_odometry_v1       <- 1.16+
    /fmu/out/vehicle_odometry_v2       <- a later ABI break
    ...

The corresponding ROS type is ``px4_msgs/msg/VehicleOdometry`` for the current
generation and ``px4_msgs_old/msg/VehicleOdometryV1`` (or similar) for a
retained one, and *which* one a given flight stack speaks depends on the PX4
tag, the ``px4_msgs`` commit and the ``dds_topics.yaml`` in between.

Hard-coding any of those three is the bug this module exists to prevent.  A
node that subscribes ``/fmu/out/vehicle_odometry`` against a 1.16 SITL gets a
subscription that is **silently never called** -- no error, no warning, just a
controller that never sees a state and a plot that stays empty.  The symmetric
failure on the publish side is worse: ``/fmu/in/vehicle_rates_setpoint`` with no
matching subscriber means the vehicle never leaves the ground while every node
reports healthy.

So nothing here is hard-coded.  :func:`select` is handed the live graph's
``(topic, [types])`` listing and picks the name **the graph actually carries**,
then imports the type string the graph reports.  It is pure Python and unit
tested without ROS; :func:`resolve` is the thin rclpy wrapper.

Selection rules, in order:

1. Only names matching ``^<base>(_v<digits>)?$`` under the node namespace.
2. Among those, prefer ones whose reported type can be imported here -- a topic
   we cannot deserialise is a mismatch to report, not a candidate to pick.
3. Among the importable ones, prefer the **highest version**, unversioned being
   version 0.  During a transition PX4 may bridge two generations at once and
   the newer is the one the current stack writes.  A tie within one version is
   broken towards ``px4_msgs`` and then towards the longer type name, so the
   pick never depends on the order the graph happened to list things in.
4. If candidates exist but none is importable, raise :class:`VersionMismatch`
   naming the wire type, the installed ``px4_msgs`` contents and the fix.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: ``vehicle_odometry_v2`` -> ("vehicle_odometry", 2); ``vehicle_odometry`` -> (..., 0)
_SUFFIX = re.compile(r"^(?P<base>.+?)(?:_v(?P<ver>\d+))?$")


class VersionMismatch(RuntimeError):
    """The graph advertises a topic whose message type we cannot import."""


class TopicNotFound(RuntimeError):
    """No topic matching the base name appeared within the timeout."""


@dataclass(frozen=True)
class TopicSpec:
    """One resolved topic: the wire name, its type string and the class."""

    topic: str          #: fully qualified name as advertised, e.g. /fmu/out/vehicle_odometry_v1
    type_str: str       #: e.g. px4_msgs/msg/VehicleOdometry
    version: int        #: 0 for unversioned
    msg_class: type | None = None

    def __str__(self):
        v = "unversioned" if self.version == 0 else f"v{self.version}"
        return f"{self.topic} [{self.type_str}, {v}]"


def split_version(name):
    """``/fmu/out/vehicle_odometry_v2`` -> ``('/fmu/out/vehicle_odometry', 2)``.

    An unversioned name gets version 0, which makes the ordering in
    :func:`select` total without a special case.
    """
    m = _SUFFIX.match(str(name))
    base, ver = m.group("base"), m.group("ver")
    return base, (int(ver) if ver is not None else 0)


def import_msg(type_str):
    """``px4_msgs/msg/VehicleOdometry`` -> the class, or None if unavailable.

    Returns None rather than raising: an un-importable type is a *candidate
    rejection* in :func:`select`, and only becomes an error if it was the only
    candidate.  That distinction is what lets a mixed graph resolve cleanly.
    """
    import importlib
    parts = str(type_str).split("/")
    if len(parts) != 3 or parts[1] != "msg":
        return None
    try:
        return getattr(importlib.import_module(f"{parts[0]}.msg"), parts[2])
    except Exception:                                      # noqa: BLE001
        return None


def select(base, graph, namespace="", importer=import_msg):
    """Pick the topic to use for ``base`` from a live-graph listing.

    ``graph`` is ``[(topic_name, [type_str, ...]), ...]`` exactly as
    ``Node.get_topic_names_and_types`` returns it.  ``base`` is the
    **unversioned** name, e.g. ``/fmu/out/vehicle_odometry``; ``namespace`` is
    prefixed to it for multi-vehicle graphs (``/px4_1``).

    Returns a :class:`TopicSpec`, or None when the base does not appear at all
    (the caller decides whether that is fatal -- for an input topic it may just
    mean the bridge has not created its subscriber yet).

    Raises :class:`VersionMismatch` when the name is there but no advertised
    type can be imported.  That is the case worth shouting about: the graph and
    ``px4_msgs`` disagree, and every downstream symptom of it is silent.
    """
    want = f"{namespace.rstrip('/')}{base}" if namespace else base
    cands, unimportable = [], []
    for topic, types in graph:
        tbase, ver = split_version(topic)
        if tbase != want:
            continue
        for ts in types:
            cls = importer(ts)
            if cls is None:
                unimportable.append((topic, ts))
            else:
                cands.append(TopicSpec(topic, ts, ver, cls))
    if cands:
        # Highest version wins.  A tie means one topic advertises two importable
        # types, which happens when an old and a new message package are both on
        # the path; prefer the canonical px4_msgs one, then take the longer
        # (more specific) name, so the pick is deterministic rather than
        # whichever the graph listed first.
        return max(cands, key=lambda c: (c.version,
                                         c.type_str.startswith("px4_msgs/"),
                                         len(c.type_str)))
    if unimportable:
        names = ", ".join(f"{t} ({ts})" for t, ts in sorted(set(unimportable)))
        raise VersionMismatch(
            f"PX4 message-version mismatch on {want!r}.\n"
            f"  the graph advertises: {names}\n"
            f"  none of those types can be imported in this Python environment.\n"
            f"  The flight stack and the installed px4_msgs are from different\n"
            f"  PX4 generations.  Rebuild px4_msgs from the SAME PX4 tag as the\n"
            f"  firmware you are flying:\n"
            f"      cd ~/px4_ws/src/px4_msgs && git fetch && git checkout <tag>\n"
            f"      cd ~/px4_ws && colcon build --packages-select px4_msgs\n"
            f"  Do NOT work around this by renaming the topic: the ABI differs,\n"
            f"  so a same-named message of the wrong generation deserialises to\n"
            f"  plausible-looking garbage.")
    return None


def graph_report(graph, namespace="", bases=None):
    """A human-readable census of the PX4 topics on the graph.

    Printed by the preflight and on any resolution failure, because the first
    question in every one of these bugs is 'what is actually being published',
    and the second is 'in which generation'.
    """
    bases = bases or DEFAULT_BASES
    lines = []
    for b in bases:
        want = f"{namespace.rstrip('/')}{b}" if namespace else b
        hits = [(t, ts) for t, tl in graph for ts in tl
                if split_version(t)[0] == want]
        if not hits:
            lines.append(f"  {want:<42} -- ABSENT")
        for t, ts in sorted(hits):
            ok = "ok " if import_msg(ts) is not None else "NO TYPE"
            lines.append(f"  {t:<42} {ts:<38} [{ok}]")
    return "\n".join(lines)


#: The topics this workspace touches.  Used by the preflight census only --
#: resolution never consults a table, it reads the graph.
DEFAULT_BASES = (
    "/fmu/out/vehicle_odometry",
    "/fmu/out/vehicle_status",
    "/fmu/out/actuator_motors",
    "/fmu/in/vehicle_rates_setpoint",
    "/fmu/in/offboard_control_mode",
    "/fmu/in/vehicle_command",
    "/fmu/in/trajectory_setpoint",
)


def resolve(node, base, timeout_s=10.0, namespace="", required=True,
            poll_s=0.25):                                  # pragma: no cover
    """rclpy wrapper around :func:`select`: spin until the topic appears.

    The graph is not populated the instant a node starts, so a single
    ``get_topic_names_and_types`` at construction time reliably misses topics
    that are about to exist.  Polling with a timeout is the difference between
    a demo that works on a cold start and one that works only on the second
    attempt.
    """
    import rclpy
    deadline = node.get_clock().now().nanoseconds * 1e-9 + float(timeout_s)
    spec = None
    while True:
        spec = select(base, node.get_topic_names_and_types(), namespace)
        if spec is not None:
            node.get_logger().info(f"px4_topics: {base} -> {spec}")
            return spec
        if node.get_clock().now().nanoseconds * 1e-9 >= deadline:
            break
        rclpy.spin_once(node, timeout_sec=poll_s)
    report = graph_report(node.get_topic_names_and_types(), namespace)
    msg = (f"px4_topics: no topic matching {base!r} (or {base}_v<N>) appeared "
           f"within {timeout_s:g} s.\nPX4 topic census:\n{report}\n"
           f"Is the uXRCE-DDS agent running, and is PX4 SITL up?\n"
           f"    MicroXRCEAgent udp4 -p 8888")
    if required:
        raise TopicNotFound(msg)
    node.get_logger().warn(msg)
    return None


def as_floats(value, n=None, name="parameter"):
    """A vector parameter as a list of floats, however ROS 2 delivered it.

    A launch file can hand a node a ``double[]`` as a genuine float list, as a
    list of strings, or as one string like ``"[0.0, 0.0, 1.5]"`` or
    ``"0.0 0.0 1.5"`` -- depending on how the value was built and whether its
    type was declared.  The failure when it arrives in the unexpected shape is
    a node that dies during construction, which from the launch output looks
    like a node that died for no reason at all.

    This lives here rather than in each node because three of them take vector
    parameters and all three were written assuming the float-list case.
    """
    import numpy as np

    if isinstance(value, str):
        parts = [p for p in value.replace("[", " ").replace("]", " ")
                 .replace(",", " ").split() if p]
    else:
        try:
            parts = list(value)
        except TypeError:
            parts = [value]
    try:
        out = [float(p) for p in parts]
    except (TypeError, ValueError) as ex:
        raise ValueError(
            f"{name} must be {n or 'a list of'} numbers; got {value!r}") from ex
    if n is not None and len(out) != n:
        raise ValueError(f"{name} must have {n} elements, got {len(out)}: {value!r}")
    return np.asarray(out, dtype=float)


def add_study_to_path():
    """Put the repository's ``study/`` directory on ``sys.path``.

    ``rdp_infer`` and ``x500_core_jax`` live there, outside the ROS workspace,
    and are imported by three packages.  Each was computing the path by
    counting ``..`` from ``__file__`` -- which is fragile, because with
    ``colcon build --symlink-install`` the module that actually runs lives
    under ``build/<pkg>/`` and the count only happens to come out the same.
    Search upward for the directory instead, and say so when it is not there:
    ``ModuleNotFoundError: No module named 'rdp_infer'`` names the symptom and
    not the cause.
    """
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    tried = []
    d = here
    for _ in range(8):
        cand = os.path.join(d, "study")
        tried.append(cand)
        if os.path.isfile(os.path.join(cand, "rdp_infer.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
        nxt = os.path.dirname(d)
        if nxt == d:
            break
        d = nxt
    raise ImportError(
        "cannot find the repository's study/ directory, which holds rdp_infer "
        "and x500_core_jax.\nSearched upward from " + here + ":\n  "
        + "\n  ".join(tried)
        + "\nThe ROS workspace is meant to sit inside the repository, beside "
          "study/.  If it was copied out, set PYTHONPATH to the study directory.")
