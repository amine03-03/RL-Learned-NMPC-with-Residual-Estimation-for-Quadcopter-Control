"""§9.12 metrics.  Pure NumPy so the offline evaluator runs without ROS.

RDP accuracy per component, adaptation time, closed-loop tracking, and real-time
timing.  The convergence criterion for T_adapt is **defined here and recorded in
the run config** -- deciding it after seeing the data is how an adaptation time
becomes whatever the author wanted.
"""
from __future__ import annotations

import numpy as np

#: (9.12) convergence criterion, fixed BEFORE the final evaluation.
ADAPT_NORM_THRESHOLD = 0.2      # normalised error below this ...
ADAPT_HOLD_S = 0.5              # ... held for at least this long
DEADLINE_MS = 20.0

CHANNELS = ("F_x", "F_y", "F_z", "M_x", "M_y", "M_z")


def rdp_accuracy(d_true, d_pred):
    """(9.6) RMSE, MAE, R^2 and NRMSE per component."""
    d_true = np.atleast_2d(np.asarray(d_true, float))
    d_pred = np.atleast_2d(np.asarray(d_pred, float))
    rows = []
    for i, name in enumerate(CHANNELS):
        e = d_pred[:, i] - d_true[:, i]
        sd = float(d_true[:, i].std())
        ss_tot = float(((d_true[:, i] - d_true[:, i].mean()) ** 2).sum())
        ss_res = float((e ** 2).sum())
        rows.append(dict(channel=name, rmse=float(np.sqrt((e ** 2).mean())),
                         mae=float(np.abs(e).mean()),
                         r2=(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                         nrmse=(float(np.sqrt((e ** 2).mean())) / sd)
                         if sd > 0 else np.nan))
    return rows


def adaptation_time(t, d_true, d_pred, t_onset, t_end=None,
                    thresh=ADAPT_NORM_THRESHOLD, hold_s=ADAPT_HOLD_S):
    """T_adapt = t_convergence - t_onset, with the criterion fixed above.

    The search window is **bounded at ``t_end``**, the instant the disturbance is
    removed.  Without that bound an estimator that never tracks the disturbance
    at all still "converges" the moment the disturbance stops, because truth and
    prediction are then both zero -- a predictor stuck at zero would score
    T_adapt = 7.0 s on a 5-12 s disturbance instead of the NaN it deserves.

    Returns NaN when the estimate never converges inside the window, which is a
    result and must be reported as one.
    """
    t = np.asarray(t, float)
    d_true = np.atleast_2d(np.asarray(d_true, float))
    d_pred = np.atleast_2d(np.asarray(d_pred, float))
    scale = np.maximum(np.abs(d_true).max(0), 1e-9)
    err = np.linalg.norm((d_pred - d_true) / scale, axis=1) / np.sqrt(d_true.shape[1])
    ok = err < thresh
    m = (t >= t_onset) if t_end is None else ((t >= t_onset) & (t < t_end))
    if not m.any():
        return float("nan")
    tt, oo = t[m], ok[m]
    dt = np.median(np.diff(tt)) if tt.size > 1 else 0.02
    need = max(int(round(hold_s / max(dt, 1e-9))), 1)
    run = 0
    for i, v in enumerate(oo):
        run = run + 1 if v else 0
        if run >= need:
            return float(tt[i - need + 1] - t_onset)
    return float("nan")


def closed_loop(t, p, p_ref, v=None, v_ref=None, tilt=None, u=None,
                t_off=None, recover_thresh=0.15):
    """Position RMSE, max error, velocity RMSE, attitude error, effort,
    constraint violations, and recovery time after the disturbance is removed."""
    p, p_ref = np.asarray(p, float), np.asarray(p_ref, float)
    e = np.linalg.norm(p - p_ref, axis=1)
    out = dict(pos_rmse=float(np.sqrt((e ** 2).mean())), pos_max=float(e.max()))
    if v is not None and v_ref is not None:
        out["vel_rmse"] = float(np.sqrt(
            (np.linalg.norm(np.asarray(v) - np.asarray(v_ref), axis=1) ** 2).mean()))
    if tilt is not None:
        out["tilt_mean"] = float(np.mean(tilt))
        out["tilt_max"] = float(np.max(tilt))
    if u is not None:
        u = np.asarray(u, float)
        out["effort"] = float((u[:, 0] ** 2).mean())
        out["sat_fraction"] = float(((u[:, 0] >= 1.0 - 1e-9)
                                     | (u[:, 0] <= 1e-9)).mean())
        out["violations"] = int(((u < -1.0 - 1e-9) | (u > 1.0 + 1e-9)).sum())
    if t_off is not None:
        t = np.asarray(t, float)
        m = t >= t_off
        if m.any():
            em, tm = e[m], t[m]
            idx = np.where(em < recover_thresh)[0]
            out["recovery_s"] = float(tm[idx[0]] - t_off) if idx.size else float("nan")
    return out


def timing_table(recorder):
    """R-T3: mean / p95 / p99 / max and the deadline-miss rate, per component."""
    return recorder.rows()


def budget_check(rows, deadline_ms=DEADLINE_MS):
    """(9.4) T_RDP + T_ACMPC + T_ROS2 < 20 ms, reported per component and total."""
    by = {r["component"]: r for r in rows}
    tot = by.get("loop")
    out = dict(deadline_ms=deadline_ms,
               components={k: dict(mean=v["mean"], p95=v["p95"], p99=v["p99"],
                                   max=v["max"], miss_rate=v["miss_rate"])
                           for k, v in by.items()})
    if tot:
        out["meets_budget_p99"] = bool(tot["p99"] < deadline_ms)
        out["miss_rate"] = tot["miss_rate"]
    return out
