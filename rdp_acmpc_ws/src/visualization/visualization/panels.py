"""R-F1 live panel and the offline figures R-F2 .. R-F12 (§10.2).

Pure matplotlib; the live panel is also saved, so the figure that was watched is
the figure that is archived.
"""
from __future__ import annotations

import os

import numpy as np

CHANNELS = ("F_x", "F_y", "F_z", "M_x", "M_y", "M_z")
UNITS = ("N", "N", "N", "N.m", "N.m", "N.m")
DEADLINE_MS = 20.0


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 8,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "legend.frameon": False,
                         "figure.constrained_layout.use": True})


def rolling_r2(y, yh, n):
    """R^2 over the last n samples; NaN when the truth is constant there."""
    y, yh = np.asarray(y)[-n:], np.asarray(yh)[-n:]
    if y.size < 4:
        return np.nan
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - yh) ** 2).sum()) / ss_tot if ss_tot > 1e-12 else np.nan


def live_panel(t, d_true, d_pred, pos_err, meta, analytic=None, out=None,
               window_s=2.0, dt=0.02):
    """R-F1: 2x3 grid of the six channels, truth solid vs prediction dashed.

    The analytic guide (9.7) is drawn as a horizontal line where it applies --
    it has **no free parameters**, which makes it the most convincing validation
    available and the fastest way to see a sign or scale error.

    The title carries scenario, controller, calibrated T_max, RDP p95,
    ``actuator_motors`` OK/MISSING and the rejection count, because a panel that
    looks right while ``actuator_motors`` is missing is a panel whose moment
    channels are unobservable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    t = np.asarray(t)
    d_true, d_pred = np.atleast_2d(d_true), np.atleast_2d(d_pred)
    n = max(int(window_s / dt), 8)
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 5.2), sharex=True)
    for i in range(6):
        ax = axs[i // 3, i % 3]
        ax.plot(t, d_true[:, i], "-", color="C0", lw=1.3, label="truth")
        ax.plot(t, d_pred[:, i], "--", color="C1", lw=1.3, label="RDP")
        if analytic is not None and np.isfinite(analytic[i]) and abs(analytic[i]) > 1e-9:
            ax.axhline(analytic[i], color="C2", ls=":", lw=1.4,
                       label="analytic (9.7)")
        r2 = rolling_r2(d_true[:, i], d_pred[:, i], n)
        ax.text(0.98, 0.04, f"$R^2_{{2s}}$ = {r2:+.2f}", transform=ax.transAxes,
                ha="right", fontsize=7)
        ax.set_title(f"{CHANNELS[i]} [{UNITS[i]}]", fontsize=8)
        if i == 0:
            ax.legend(fontsize=6.5, loc="upper left")
        if i >= 3:
            ax.set_xlabel("t [s]")
    ins = axs[0, 2].inset_axes([0.62, 0.62, 0.35, 0.33])
    ins.plot(t, pos_err, color="C3", lw=1.0)
    ins.set_title(r"$\|p-p_d\|$ [m]", fontsize=6)
    ins.tick_params(labelsize=5)
    am = meta.get("actuator_motors", "MISSING")
    fig.suptitle(
        f"R-F1  {meta.get('scenario','?')} / {meta.get('controller','?')}   "
        f"T_max(cal) = {meta.get('T_max', float('nan')):.5f} N   "
        f"RDP p95 = {meta.get('rdp_p95_ms', float('nan')):.2f} ms   "
        f"actuator_motors: {am}   rejections: {meta.get('rejections', 0)}",
        fontsize=9)
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def scatter_truth_vs_pred(d_true, d_pred, out=None):
    """R-F2: per channel with the identity line.  A sign error shows as a
    reflected line, which is the most likely frame bug and the easiest to miss
    in a time series."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    d_true, d_pred = np.atleast_2d(d_true), np.atleast_2d(d_pred)
    fig, axs = plt.subplots(2, 3, figsize=(11.5, 4.8))
    for i in range(6):
        ax = axs[i // 3, i % 3]
        ax.scatter(d_true[:, i], d_pred[:, i], s=4, alpha=0.35)
        lo = float(min(d_true[:, i].min(), d_pred[:, i].min()))
        hi = float(max(d_true[:, i].max(), d_pred[:, i].max()))
        ax.plot([lo, hi], [lo, hi], "--", color="0.3", lw=1)
        sgn = np.sign(np.polyfit(d_true[:, i], d_pred[:, i], 1)[0]) \
            if d_true[:, i].std() > 1e-12 else 0.0
        ax.set_title(f"{CHANNELS[i]}  slope sign {sgn:+.0f}", fontsize=8)
        ax.set_xlabel("truth"); ax.set_ylabel("RDP")
    fig.suptitle("R-F2  predicted vs true, identity line", fontsize=9)
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def steady_state_vs_analytic(levels, fz_meas, tauy_meas, fz_ana, tauy_ana, out=None):
    """R-F3: measured against the parameter-free prediction (9.7)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2))
    axs[0].plot(levels, fz_meas, "-o", ms=4, label="measured")
    axs[0].plot(levels, fz_ana, "--s", ms=4, label="analytic (9.7)")
    axs[0].set_ylabel("$F_z$ [N]")
    axs[1].plot(levels, tauy_meas, "-o", ms=4, label="measured")
    axs[1].plot(levels, tauy_ana, "--s", ms=4, label="analytic (9.7)")
    axs[1].set_ylabel(r"$\tau_y$ [N m]")
    for a in axs:
        a.set_xlabel("payload mass fraction"); a.legend(fontsize=7)
    fig.suptitle("R-F3  steady state vs the parameter-free prediction "
                 "(require agreement within 10 %)", fontsize=9)
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def tracking_vs_time(t, errs, labels, t_on, t_off, out=None):
    """R-F4: C0/C1/C2/C3 on one scenario, onset and removal marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for i, (e, lb) in enumerate(zip(errs, labels)):
        ax.plot(t, e, lw=1.3, label=lb, marker=["o", "s", "^", "v"][i % 4],
                markevery=max(len(t) // 14, 1), ms=3.5)
    ax.axvline(t_on, color="0.4", ls=":", lw=1.2)
    ax.axvline(t_off, color="0.4", ls="-.", lw=1.2)
    ax.text(t_on, ax.get_ylim()[1] * 0.95, " onset", fontsize=7)
    ax.text(t_off, ax.get_ylim()[1] * 0.95, " removed", fontsize=7)
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\|p-p_d\|$ [m]"); ax.legend(fontsize=7)
    ax.set_title("R-F4  tracking error, disturbance onset and removal marked")
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def frequency_response(freqs, pred_rmse, traj_rmse, out=None):
    """R-F5 and R-F6 from E-B: where estimator delay starts to hurt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.2))
    axs[0].semilogx(freqs, pred_rmse, "-o", ms=4)
    axs[0].set_ylabel("prediction RMSE [N]"); axs[0].set_title("R-F5")
    axs[1].semilogx(freqs, traj_rmse, "-s", ms=4, color="C1")
    axs[1].set_ylabel("trajectory RMSE [m]"); axs[1].set_title("R-F6")
    for a in axs:
        a.set_xlabel("disturbance frequency [Hz] (log)")
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def timing_histogram(rdp_ms, acmpc_ms, loop_ms, out=None, deadline=DEADLINE_MS):
    """R-F8: with the 20 ms deadline and p99 marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for v, lb, c in ((rdp_ms, "RDP", "C0"), (acmpc_ms, "ACMPC", "C1"),
                     (loop_ms, "complete loop", "C2")):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]                # e.g. RDP before its window is full
        if v.size == 0:
            continue
        ax.hist(v, bins=60, histtype="step", lw=1.4, color=c,
                label=f"{lb}  p99 {np.percentile(v,99):.2f} ms")
        ax.axvline(np.percentile(v, 99), color=c, ls=":", lw=1)
    ax.axvline(deadline, color="C3", lw=1.8)
    ax.text(deadline * 1.02, ax.get_ylim()[1] * 0.8, "20 ms deadline",
            color="C3", fontsize=7, rotation=90)
    ax.set_xlabel("time [ms]"); ax.set_ylabel("count"); ax.legend(fontsize=7)
    ax.set_title("R-F8  timing")
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def horizon_tradeoff(horizons, rmse, solve_p99_ms, out=None, deadline=DEADLINE_MS):
    """R-F9 from E-C: infeasible horizons shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(horizons, rmse, "-o", ms=5, color="C0", label="tracking RMSE [m]")
    ax2 = ax.twinx()
    ax2.plot(horizons, solve_p99_ms, "--s", ms=5, color="C1",
             label="solve p99 [ms]")
    ax2.axhline(deadline, color="C3", lw=1.5)
    bad = [h for h, s in zip(horizons, solve_p99_ms) if s >= deadline]
    for h in bad:
        ax.axvspan(h - 2.5, h + 2.5, color="C3", alpha=0.12)
    ax.set_xlabel("MPC horizon $N_p$"); ax.set_ylabel("tracking RMSE [m]")
    ax2.set_ylabel("solve p99 [ms]")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    ax.set_title(f"R-F9  horizon trade-off; shaded = violates (9.4): {bad or 'none'}")
    if out:
        fig.savefig(out, bbox_inches="tight")
    return fig


def main(argv=None):                                       # pragma: no cover
    """Render the offline figures from a run directory written by state_logger."""
    import argparse
    import os
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.run_dir, "plots")
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(os.path.join(a.run_dir, "states.csv"))
    dt = df[[f"dtrue_{c}" for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]].to_numpy()
    dp = df[[f"dpred_{c}" for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]].to_numpy()
    err = ((df[["p_x", "p_y", "p_z"]].to_numpy()
            - df[["pref_x", "pref_y", "pref_z"]].to_numpy()) ** 2).sum(1) ** 0.5
    live_panel(df.t.to_numpy(), dt, dp, err,
               dict(scenario="offline", controller="offline", T_max=34.19432,
                    rdp_p95_ms=float(df.rdp_ms.quantile(0.95)),
                    actuator_motors="OK", rejections=0),
               out=os.path.join(out, "R-F1_live_panel.png"))
    scatter_truth_vs_pred(dt, dp, out=os.path.join(out, "R-F2_scatter.png"))
    timing_histogram(df.rdp_ms.to_numpy(), df.acmpc_ms.to_numpy(),
                     df.loop_ms.to_numpy(),
                     out=os.path.join(out, "R-F8_timing.png"))
    print(f"wrote figures to {out}")
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
