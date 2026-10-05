"""Layout of ``/acmpc/status`` -- what the controller saw and did at one tick.

Published by ``controller_node`` every tick as a ``std_msgs/Float64MultiArray``
and consumed by ``rdp_estimator`` and ``state_logger``.  Carrying the state,
the reference and the command **of the same tick** in one message is what keeps
the three nodes aligned: the estimator's frame (6.2) needs ``p - p_ref`` and
``u_{t-1} - u_ref`` evaluated on the controller's clock, and a separately
clocked reference node cannot provide that.

Everything is ENU/FLU and in the study's normalised CTBR units
(``u = [c, w_x/10, w_y/10, w_z/4]``).  It carries no ground truth (§9.1).

Pure NumPy, importable with no ROS on the path.
"""
from __future__ import annotations

import numpy as np

#: (name, width) in message order.
FIELDS = (
    ("t", 1),          # controller time since its first odometry sample [s]
    ("p", 3),          # position used by the solve
    ("v", 3),          # velocity
    ("q", 4),          # attitude, scalar-first, body FLU -> world ENU
    ("om", 3),         # body rate FLU [rad/s]
    ("p_ref", 3),      # reference position at t
    ("v_ref", 3),      # reference velocity at t
    ("q_ref", 4),      # reference attitude at t
    ("u_prev", 4),     # command applied BEFORE this tick (frame (6.2) uses it)
    ("u_ref", 4),      # reference input at t, (4.8)
    ("u", 4),          # command computed at this tick and sent to PX4
    ("solve_ms", 1),   # wall time of the solve [ms]
    ("loop_ms", 1),    # wall time of the whole tick, odometry -> setpoints sent [ms]
)
SLICES = {}
_i = 0
for _name, _w in FIELDS:
    SLICES[_name] = slice(_i, _i + _w)
    _i += _w
SIZE = _i              # 38


def pack(**kw):
    """Keyword arguments named as in :data:`FIELDS` -> flat list of floats."""
    out = np.zeros(SIZE)
    for name, w in FIELDS:
        if name not in kw:
            raise KeyError(f"status field {name!r} missing")
        out[SLICES[name]] = np.asarray(kw[name], float).reshape(w)
    return out.tolist()


def unpack(data):
    """Flat sequence -> dict of arrays (scalars for width-1 fields)."""
    a = np.asarray(data, float)
    if a.size != SIZE:
        raise ValueError(f"/acmpc/status must have {SIZE} values, got {a.size}")
    d = {name: a[SLICES[name]].copy() for name, _ in FIELDS}
    d["t"] = float(d["t"][0])
    d["solve_ms"] = float(d["solve_ms"][0])
    d["loop_ms"] = float(d["loop_ms"][0])
    return d
