"""Causal history ring buffer for the RDP (§9.1, §9.5).

The buffer is the **only** path from the vehicle to the predictor's input, and
the disturbance manager has no handle on it.  That is how the causality
constraint of §9.1 is enforced in code rather than by convention: there is no
call by which an injected disturbance value could reach a window.

Frame layout is (6.2), 26-D, with vec(R) **row-major**:

    [ p - p_ref (3) | vec(R) (9) | v (3) | omega (3) || u_{t-1} - u_ref (4) || PWM (4) ]
"""
from __future__ import annotations

import numpy as np

FRAME_DIM = 26


def build_frame(p, p_ref, R, v, omega, u_prev, u_ref, pwm):
    """Assemble one 26-D frame.  All inputs are ENU/FLU, already converted."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    f = np.empty(FRAME_DIM)
    f[0:3] = np.asarray(p, float) - np.asarray(p_ref, float)
    f[3:12] = R.reshape(9)                       # row-major, as specified
    f[12:15] = np.asarray(v, float)
    f[15:18] = np.asarray(omega, float)
    f[18:22] = np.asarray(u_prev, float) - np.asarray(u_ref, float)
    f[22:26] = np.asarray(pwm, float)
    return f


class RingBuffer:
    """Fixed-length causal history.  ``ready()`` is False until H real frames
    have been pushed; before that the window is zero-padded and the estimate is
    forced to zero rather than being taken from padding."""

    def __init__(self, H, dim=FRAME_DIM):
        self.H = int(H)
        self.dim = int(dim)
        self._buf = np.zeros((self.H, self.dim))
        self._n = 0
        self._stamps = np.zeros(self.H)

    def push(self, frame, stamp=0.0):
        f = np.asarray(frame, dtype=float).reshape(-1)
        if f.size != self.dim:
            raise ValueError(f"frame must be {self.dim}-D, got {f.size}")
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = f
        self._stamps = np.roll(self._stamps, -1)
        self._stamps[-1] = stamp
        self._n = min(self._n + 1, self.H)

    def window(self):
        """(1, H, dim) -- ready for ``rdp_infer.RDP.predict``."""
        return self._buf[None].copy()

    def ready(self):
        return self._n >= self.H

    def span_s(self):
        return float(self._stamps[-1] - self._stamps[0]) if self.ready() else 0.0

    def reset(self):
        self._buf[:] = 0.0
        self._stamps[:] = 0.0
        self._n = 0
