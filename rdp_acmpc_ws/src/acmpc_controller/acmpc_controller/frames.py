"""frames.py -- PX4 <-> study frame conversions and the CTBR exit mapping.

Pure NumPy, importable with no ROS on the path, so §11 T-18 can run in the study
test suite.  **Everything inside the package is ENU/FLU; PX4 is NED/FRD on the
wire** and the conversion happens at this boundary and nowhere else.

A green control-law test says nothing about frames.  Keep them separate.
"""
from __future__ import annotations

import numpy as np

#: q_{NED->ENU}: the rotation taking a NED world vector to ENU.  It is the
#: 180 deg rotation about the axis (1,1,0)/sqrt(2): x<->y, z flips.
Q_NED_ENU = np.array([0.0, np.sqrt(0.5), np.sqrt(0.5), 0.0])
#: q_{FRD->FLU}: 180 deg about body x: y and z flip.
Q_FRD_FLU = np.array([0.0, 1.0, 0.0, 0.0])


def qmul(a, b):
    """Hamilton product, scalar-first, (...,4)."""
    a, b = np.asarray(a), np.asarray(b)
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return np.concatenate([aw * bw - (av * bv).sum(-1, keepdims=True),
                           aw * bv + bw * av + np.cross(av, bv)], -1)


def qconj(q):
    return np.asarray(q) * np.array([1.0, -1.0, -1.0, -1.0])


def qrot(q, v):
    """Rotate v by q (body->world for a body->world quaternion)."""
    q, v = np.asarray(q), np.asarray(v)
    z = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
    return qmul(qmul(q, z), qconj(q))[..., 1:]


def qrotmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def ned_to_enu_vec(v):
    """(N,E,D) -> (E,N,U).  Involutive."""
    v = np.asarray(v)
    return np.stack([v[..., 1], v[..., 0], -v[..., 2]], -1)


def frd_to_flu_vec(v):
    """(F,R,D) -> (F,L,U).  Involutive."""
    v = np.asarray(v)
    return np.stack([v[..., 0], -v[..., 1], -v[..., 2]], -1)


enu_to_ned_vec = ned_to_enu_vec
flu_to_frd_vec = frd_to_flu_vec


def px4_quat_to_enu_flu(q_px4):
    """q_{ENU/FLU} = q_{NED->ENU} (x) q_{PX4} (x) q_{FRD->FLU}  (§9.4)."""
    return qnormalise(qmul(qmul(Q_NED_ENU, q_px4), Q_FRD_FLU))


def enu_flu_to_px4_quat(q):
    return qnormalise(qmul(qmul(qconj(Q_NED_ENU), q), qconj(Q_FRD_FLU)))


def qnormalise(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def ctbr_to_px4_rates(om_flu):
    """CTBR body rates ENU/FLU -> the FRD fields of ``VehicleRatesSetpoint``.

        roll = om_x,  pitch = -om_y,  yaw = -om_z          (§9.4)
    """
    om = np.asarray(om_flu)
    return np.stack([om[..., 0], -om[..., 1], -om[..., 2]], -1)


def ctbr_to_px4_thrust(collective):
    """``thrust_body`` is FRD and down-positive, so a positive collective is -z."""
    c = np.asarray(collective)
    return np.stack([np.zeros_like(c), np.zeros_like(c), -c], -1)


#: ``px4_msgs/VehicleOdometry.velocity_frame`` values.
VELOCITY_FRAME_NED = 1
VELOCITY_FRAME_BODY_FRD = 3


def odometry_to_enu_flu(position, q, velocity, angular_velocity,
                        velocity_frame=VELOCITY_FRAME_NED):
    """``VehicleOdometry`` fields -> (p, v, q, om) in ENU/FLU.

    PX4 publishes position and (by default) velocity in NED, the attitude as
    the FRD->NED quaternion and the body rate in FRD.  A body-frame velocity is
    rotated to the world with the converted attitude.  Any other velocity frame
    is refused rather than guessed.
    """
    p = ned_to_enu_vec(np.asarray(position, float))
    qe = px4_quat_to_enu_flu(np.asarray(q, float))
    if int(velocity_frame) == VELOCITY_FRAME_NED:
        v = ned_to_enu_vec(np.asarray(velocity, float))
    elif int(velocity_frame) == VELOCITY_FRAME_BODY_FRD:
        v = qrot(qe, frd_to_flu_vec(np.asarray(velocity, float)))
    else:
        raise ValueError(f"unsupported VehicleOdometry.velocity_frame "
                         f"{velocity_frame}; expected NED (1) or BODY_FRD (3)")
    om = frd_to_flu_vec(np.asarray(angular_velocity, float))
    return p, v, qe, om


def tilt_angle(q):
    """Angle between the body-up axis and world up.  Heading-independent."""
    zb = qrotmat(np.atleast_2d(q))[..., :, 2]
    return np.arccos(np.clip(zb[..., 2], -1.0, 1.0))


class QuaternionDifferentiator:
    """omega_dot source of last resort (§9.4).

    Uses ``timestamp_sample`` and **not** the wall clock, keeps the quaternion
    hemisphere continuous, floors dt at 2 ms, rejects |omega| > 12 rad/s and
    low-passes at 15 Hz.  ``rejections`` is counted and shown on the live panel:
    a drained queue can otherwise produce |omega| = 15 against a true 0.4.
    """

    DT_FLOOR = 2e-3
    OM_REJECT = 12.0
    F_LP = 15.0

    def __init__(self):
        self.q_prev = None
        self.t_prev = None
        self.om = np.zeros(3)
        self.rejections = 0
        self.n = 0

    def update(self, q, timestamp_sample_s):
        q = qnormalise(np.asarray(q, dtype=float))
        if self.q_prev is not None and float(q @ self.q_prev) < 0.0:
            q = -q                                   # hemisphere continuity
        if self.q_prev is None:
            self.q_prev, self.t_prev = q, timestamp_sample_s
            return self.om.copy()
        dt = max(timestamp_sample_s - self.t_prev, self.DT_FLOOR)
        dq = qmul(qconj(self.q_prev), q)
        om_raw = 2.0 * dq[1:] / dt
        self.n += 1
        if np.linalg.norm(om_raw) > self.OM_REJECT:
            self.rejections += 1                     # keep the last good value
        else:
            a = np.exp(-2 * np.pi * self.F_LP * dt)
            self.om = a * self.om + (1 - a) * om_raw
        self.q_prev, self.t_prev = q, timestamp_sample_s
        return self.om.copy()
