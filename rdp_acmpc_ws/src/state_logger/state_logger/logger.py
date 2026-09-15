"""50 Hz synchronised CSV writer (§9.6, §9.14).

Per sample: timestamp, state, reference, ACMPC command, motor commands, the
**true** disturbance and the prediction.  Ground truth is written here and
nowhere else on the online path -- §9.1 keeps it away from the controller.

Episodes are written with an explicit ``episode`` column so the training split
can be **by complete episode, never by shuffled samples**, and so windows can be
dropped where they would straddle a boundary.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

COLUMNS = (["t", "episode", "phase"]
           + [f"p_{a}" for a in "xyz"] + [f"v_{a}" for a in "xyz"]
           + [f"q_{a}" for a in "wxyz"] + [f"om_{a}" for a in "xyz"]
           + [f"pref_{a}" for a in "xyz"] + [f"vref_{a}" for a in "xyz"]
           + ["u_c", "u_wx", "u_wy", "u_wz"]
           + [f"pwm_{i}" for i in range(4)]
           + [f"dtrue_{c}" for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
           + [f"dpred_{c}" for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
           + ["rdp_ms", "acmpc_ms", "loop_ms", "rdp_fault", "rdp_ready"])


class StateLogger:
    def __init__(self, run_dir, name="states.csv"):
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, name)
        self._fh = open(self.path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(COLUMNS)
        self.n = 0

    def log(self, t, episode, phase, p, v, q, om, p_ref, v_ref, u, pwm,
            d_true, d_pred, rdp_ms=np.nan, acmpc_ms=np.nan, loop_ms=np.nan,
            rdp_fault=False, rdp_ready=True):
        row = ([f"{t:.6f}", int(episode), str(phase)]
               + [f"{z:.9g}" for z in np.asarray(p, float).ravel()]
               + [f"{z:.9g}" for z in np.asarray(v, float).ravel()]
               + [f"{z:.9g}" for z in np.asarray(q, float).ravel()]
               + [f"{z:.9g}" for z in np.asarray(om, float).ravel()]
               + [f"{z:.9g}" for z in np.asarray(p_ref, float).ravel()]
               + [f"{z:.9g}" for z in np.asarray(v_ref, float).ravel()]
               + [f"{z:.9g}" for z in np.asarray(u, float).ravel()]
               + [f"{z:.9g}" for z in np.asarray(pwm, float).ravel()]
               + [f"{z:.9g}" for z in np.asarray(d_true, float).ravel()]
               + [f"{z:.9g}" for z in np.asarray(d_pred, float).ravel()]
               + [f"{rdp_ms:.4f}", f"{acmpc_ms:.4f}", f"{loop_ms:.4f}",
                  int(bool(rdp_fault)), int(bool(rdp_ready))])
        self._w.writerow(row)
        self.n += 1
        if self.n % 50 == 0:
            self._fh.flush()

    def close(self):
        self._fh.flush()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def split_frames(df, H):
    """Windows that never straddle an episode boundary (§9.6)."""
    import pandas as pd
    out_w, out_y = [], []
    fcols = [c for c in df.columns if c.startswith(("p_", "v_", "q_", "om_", "u_", "pwm_"))]
    ycols = [c for c in df.columns if c.startswith("dtrue_")]
    for _, g in df.groupby("episode"):
        F = g[fcols].to_numpy()
        Y = g[ycols].to_numpy()
        for k in range(H - 1, len(g)):
            out_w.append(F[k - H + 1:k + 1])
            out_y.append(Y[k])
    if not out_w:
        return np.zeros((0, H, len(fcols))), np.zeros((0, len(ycols)))
    return np.stack(out_w), np.stack(out_y)
