"""Rebuild every report figure from the companion CSVs -- no training required.

Each notebook writes ``artifacts/figures/F*.png`` **and** a companion
``F*.csv`` holding exactly the data it plotted.  This script reads only those
CSVs, so the whole figure set can be regenerated, restyled or repaired in
seconds instead of re-running a multi-hour campaign.  Three things it is for:

  1. a notebook cell that crashed leaves a hole -- fill it without a retrain;
  2. restyling for the report (one palette, one grid, one font) in one place;
  3. the report becomes reproducible from committed CSVs alone.

A figure whose CSV is absent is SKIPPED and named, never faked: `report.tex`'s
``\\realfig`` already renders a labelled placeholder for a missing file, and a
silently invented figure is worse than a visible gap.

    python make_report_figures.py                # all of them
    python make_report_figures.py --only F9 F21  # just these
    python make_report_figures.py --list         # what is buildable right now

Style notes, from the data-viz method:
  * categorical hues are assigned in a FIXED slot order, never cycled, and the
    order is the validated one (worst adjacent CVD dE 9.1, normal-vision 19.6);
  * no figure uses two y-scales -- where a notebook used ``twinx`` the rebuild
    uses two panels, because a dual axis lets the author choose the crossing;
  * every series carries a marker or dash as well as a hue, so the set survives
    greyscale printing and colour-vision deficiency;
  * three hues sit below 3:1 on white, so the method's relief rule applies: the
    companion CSV beside each PNG *is* the table view that discharges it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x500_core_jax as X                                        # noqa: E402

FIG = X.apath("figures")

#: Validated categorical order (see references/palette.md).  Assigned by slot,
#: never cycled: a 9th series folds into "other" or becomes small multiples.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7",
     "#008300", "#e34948"]
#: Secondary encoding, so identity never rests on hue alone.
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
DASH = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), "-", "--"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a84"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e6e6e2", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.6,
    "lines.markersize": 3.6, "figure.facecolor": "white",
})


def _tidy(ax, title=None, xl=None, yl=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, color=INK, loc="left")
    if xl:
        ax.set_xlabel(xl)
    if yl:
        ax.set_ylabel(yl)
    return ax


def _csv(name):
    p = os.path.join(FIG, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.png"))
    plt.close(fig)


def _barcmp(ax, df, idx, cols, ylabel, title, logy=False):
    """Grouped bars, one hue per column, with the value written on each bar.

    The written value is the relief the method requires for the low-contrast
    hues, and it removes the need to read a height off a gridline.
    """
    xs = np.arange(len(df))
    w = 0.8 / max(len(cols), 1)
    # three or more series make the bars narrow, so the value labels are set
    # vertically: they are the relief the low-contrast hues require, and they
    # must not collide with their neighbours to provide it
    rot = 90 if len(cols) >= 3 else 0
    for j, c in enumerate(cols):
        v = pd.to_numeric(df[c], errors="coerce").values
        ax.bar(xs + (j - (len(cols) - 1) / 2) * w, v, w * 0.92,
               label=str(c).replace("rmse_", "").replace("_", " "),
               color=C[j % len(C)], edgecolor="white", linewidth=0.5)
        for x, y in zip(xs + (j - (len(cols) - 1) / 2) * w, v):
            if np.isfinite(y):
                ax.annotate(f"{y:.3g}", (x, y), ha="center", va="bottom",
                            fontsize=5.0, color=INK2, xytext=(0, 1.6),
                            textcoords="offset points", rotation=rot)
    if rot:                      # headroom for the upright labels
        ax.margins(y=0.22)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in df[idx]], rotation=20, ha="right",
                       fontsize=6)
    if logy:
        ax.set_yscale("log")
    _tidy(ax, title, None, ylabel)
    if len(cols) >= 2:
        ax.legend(ncol=min(len(cols), 4))


# --------------------------------------------------------------------------- #
# Notebook 1 -- the control problem
# --------------------------------------------------------------------------- #
def F1_feasibility(d):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for j, k in enumerate(sorted(d.kind.unique())):
        s = d[d.kind == k].sort_values("R")
        axs[0].plot(s.R, s.peak_a_uncapped, ls=DASH[j % len(DASH)], color=C[j % len(C)],
                    marker=MK[j % len(MK)], markevery=6, label=k)
    axs[0].axhline(X.A_LAT_MAX, color=INK2, lw=0.9, ls=(0, (4, 2)))
    axs[0].annotate(f"budget {X.A_LAT_MAX:.2f}", (d.R.min(), X.A_LAT_MAX),
                    xytext=(1, 2), textcoords="offset points", fontsize=5.6, color=INK2)
    axs[0].set_yscale("log")
    _tidy(axs[0], "(a) demand before the cap", "radius R [m]", r"peak $|a|$ [m/s$^2$]")
    axs[0].legend(ncol=2)
    frac = d.groupby("kind").capped.mean().sort_values()
    axs[1].barh(range(len(frac)), frac.values * 100,
                color=[C[0] if v == 0 else C[1] for v in frac.values],
                edgecolor="white", linewidth=0.5)
    for i, v in enumerate(frac.values * 100):
        axs[1].annotate(f"{v:.0f}%", (v, i), xytext=(2, -1.6),
                        textcoords="offset points", fontsize=6, color=INK2)
    axs[1].set_yticks(range(len(frac)))
    axs[1].set_yticklabels(frac.index)
    axs[1].set_xlim(0, 108)
    _tidy(axs[1], "(b) radii the cap binds on", "capped [%]")
    _save(fig, "F1_feasibility")


def F2_lqr_feedforward(d):
    p = d.pivot_table(index="path", columns="ctrl", values="rmse").reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    _barcmp(ax, p, "path", [c for c in p.columns if c != "path"],
            "position RMSE [m]", "F2  the feed-forward term alone")
    _save(fig, "F2_lqr_feedforward")


def F3_horizon(d):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    pairs = [("circle", "circle_frozen"), ("fig8", "fig8_frozen"),
             ("square", "square_frozen")]
    for j, (pv, fr) in enumerate(pairs):
        if pv not in d:
            continue
        axs[0].plot(d.N, d[pv], ls=DASH[0], color=C[j], marker=MK[j], label=f"{pv}, preview")
        axs[0].plot(d.N, d[fr], ls=DASH[1], color=C[j], marker=MK[j], mfc="white",
                    alpha=0.85, label=f"{pv}, frozen")
    _tidy(axs[0], "(a) preview vs frozen reference", "horizon N", "RMSE [m]")
    axs[0].legend(ncol=2, fontsize=5.6)
    # latency is a different measure from RMSE -- its OWN panel, never a twin axis
    axs[1].plot(d.N, d.ms_median, "-", color=C[0], marker="o", label="median")
    axs[1].plot(d.N, d.ms_p95, "--", color=C[1], marker="s", label="p95")
    axs[1].axhline(20.0, color=INK2, lw=0.9, ls=(0, (4, 2)))
    axs[1].annotate("20 ms control period", (d.N.min(), 20.0), xytext=(1, 2),
                    textcoords="offset points", fontsize=5.6, color=INK2)
    axs[1].set_yscale("log")
    _tidy(axs[1], "(b) single-vehicle solve latency", "horizon N", "ms")
    axs[1].legend()
    _save(fig, "F3_horizon")


def F3b_noise(d):
    p = d.pivot_table(index="noise", columns="ctrl", values="rmse").reindex(
        [n for n in ("off", "low", "high") if n in set(d.noise)]).reset_index()
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    _barcmp(ax, p, "noise", [c for c in p.columns if c != "noise"],
            "RMSE [m]", "F3b  degradation with measurement noise")
    _save(fig, "F3b_noise")


def F4_corner(d):
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot(d.x1, d.y1, "-", color=C[0], lw=1.5, label="N = 1")
    ax.plot(d.x10, d.y10, "--", color=C[1], lw=1.5, label="N = 10")
    ax.set_aspect("equal")
    _tidy(ax, "F4  the corner, N=1 vs N=10", "x [m]", "y [m]")
    ax.legend()
    _save(fig, "F4_corner")


def F5_qr_heatmap(d):
    piv = d.pivot_table(index="Q_pos", columns="R_rate", values="rmse")
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    # sequential magnitude -> ONE hue, light to dark
    im = ax.imshow(piv.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", SEQ), origin="lower", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{v:g}" for v in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{v:g}" for v in piv.index])
    best = np.unravel_index(np.nanargmin(piv.values), piv.values.shape)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.annotate(f"{v:.3f}", (j, i), ha="center", va="center", fontsize=5.4,
                            color="white" if v > np.nanmedian(piv.values) else INK)
    ax.add_patch(plt.Rectangle((best[1] - .5, best[0] - .5), 1, 1, fill=False,
                               edgecolor=C[1], lw=1.6))
    ax.grid(False)
    _tidy(ax, "F5  Q/R sweep (argmin boxed)", r"$R_\mathrm{rate}$", r"$Q_\mathrm{pos}$")
    fig.colorbar(im, ax=ax, label="RMSE [m]", fraction=0.046)
    _save(fig, "F5_qr_heatmap")


# --------------------------------------------------------------------------- #
# Notebook 2 -- LLTC
# --------------------------------------------------------------------------- #
def F6_lltc_fit(d):
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.scatter(d.realised, d.predicted, s=7, alpha=0.45, color=C[0],
               edgecolor="none", label="candidate")
    lo = float(min(d.realised.min(), d.predicted.min()))
    hi = float(max(d.realised.max(), d.predicted.max()))
    ax.plot([lo, hi], [lo, hi], "--", color=INK2, lw=0.9, label="identity")
    ss = float(((d.realised - d.predicted) ** 2).sum())
    st = float(((d.realised - d.realised.mean()) ** 2).sum())
    ax.annotate(f"$R^2$ = {1 - ss / max(st, 1e-30):+.4f}", (0.04, 0.93),
                xycoords="axes fraction", fontsize=7, color=INK)
    _tidy(ax, "F6  fitted vs realised cost-to-go", r"realised $V_1$", r"predicted")
    ax.legend(loc="lower right")
    _save(fig, "F6_lltc_fit")


def F7_lltc_locality(d):
    # reach and acceptance are different measures: two panels, not a twin axis
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.4))
    axs[0].plot(d.Q_scale, d.reach_m, "-", color=C[0], marker="o")
    _tidy(axs[0], "(a) terminal-set reach", r"$Q_p$ scale", "reach [m]")
    axs[1].plot(d.Q_scale, d.acceptance, "--", color=C[1], marker="s")
    axs[1].set_ylim(0, 1.05)
    _tidy(axs[1], "(b) acceptance", r"$Q_p$ scale", "accepted fraction")
    _save(fig, "F7_lltc_locality")


def F8_lltc_spectrum(d):
    sl = d.iloc[:, 0].values
    ev = d.iloc[:, 1:].values
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i in range(ev.shape[1]):
        ax.semilogy(sl, np.maximum(ev[:, i], 1e-12), color=C[i % len(C)],
                    ls=DASH[i % len(DASH)], marker=MK[i % len(MK)], markevery=14,
                    label=rf"$\lambda_{{{i}}}$")
    _tidy(ax, r"F8  spectrum of $P_\theta$ along an error slice",
          "$e_x$ [m]", r"eig $P_\theta(e)$")
    ax.legend(ncol=3, fontsize=5.6)
    _save(fig, "F8_lltc_spectrum")


# --------------------------------------------------------------------------- #
# Notebook 3 -- AC-MPC
# --------------------------------------------------------------------------- #
def F9_representations(d):
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    for j, col in enumerate(d.columns):
        rep = str(col).split("_")[0]
        slot = {"diag": 0, "chol": 1, "full": 2}.get(rep, j)
        ax.plot(np.arange(len(d)), d[col], ls=DASH[slot], color=C[slot],
                marker=MK[slot], markevery=max(len(d) // 6, 1), alpha=0.9,
                label=str(col).replace("_", "/"))
    _tidy(ax, "F9  learning curves by representation", "iteration", "mean reward")
    ax.legend(ncol=3, fontsize=5.6)
    _save(fig, "F9_representations")


def F10_exploration(d):
    p = d.pivot_table(index="sigma", columns="arch", values="rmse").reset_index()
    q = d.pivot_table(index="sigma", columns="arch", values="train_sat").reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
    _barcmp(axs[0], p, "sigma", [c for c in p.columns if c != "sigma"],
            "RMSE [m]", r"(a) accuracy vs $\sigma$")
    _barcmp(axs[1], q, "sigma", [c for c in q.columns if c != "sigma"],
            "train saturation [-]", r"(b) the mechanism: saturation vs $\sigma$")
    axs[1].axhline(0.05, color=INK2, lw=0.9, ls=(0, (4, 2)))
    axs[1].annotate("5 % gate", (0, 0.05), xytext=(1, 2), textcoords="offset points",
                    fontsize=5.6, color=INK2)
    _save(fig, "F10_exploration")


def F11_mpve(d):
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for j, flag in enumerate(sorted(d.mpve.unique())):
        s = d[d.mpve == flag].sort_values("lam")
        axs[0].plot(s.lam, s.final_reward, ls=DASH[j], color=C[j], marker=MK[j],
                    label=f"MPVE {'on' if flag else 'off'}")
        axs[1].plot(s.lam, s.value_loss, ls=DASH[j], color=C[j], marker=MK[j],
                    label=f"MPVE {'on' if flag else 'off'}")
    _tidy(axs[0], r"(a) final reward vs $\lambda$", r"$\lambda$", "reward")
    axs[0].legend()
    axs[1].set_yscale("log")
    _tidy(axs[1], "(b) critic loss", r"$\lambda$", "value loss")
    axs[1].legend()
    _save(fig, "F11_mpve")


def F12_interpretability(d):
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    xs = np.arange(len(d))
    ax.bar(xs - 0.2, d.hand, 0.38, color=C[0], label="hand-chosen",
           edgecolor="white", linewidth=0.5)
    ax.bar(xs + 0.2, d.learned, 0.38, color=C[1], label="learned",
           edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(d.channel, rotation=35, ha="right", fontsize=6)
    _tidy(ax, "F12  learned stage weights against the hand ones", None, "weight")
    ax.legend()
    _save(fig, "F12_interpretability")


# --------------------------------------------------------------------------- #
# Notebook 4 -- domain randomisation
# --------------------------------------------------------------------------- #
def F13_axis_competence(d):
    axes_ = sorted(d.axis.unique())
    fig, axs = plt.subplots(1, len(axes_), figsize=(2.1 * len(axes_), 2.4),
                            sharey=True)
    axs = np.atleast_1d(axs)
    pols = sorted(d.policy.unique())
    for a, axn in zip(axs, axes_):
        for j, pol in enumerate(pols):
            s = d[(d.axis == axn) & (d.policy == pol)].sort_values("factor_realised")
            a.plot(s.factor_realised, s.rmse, ls=DASH[j % len(DASH)], color=C[j % len(C)],
                   marker=MK[j % len(MK)], label=pol)
        _tidy(a, f"$\\lambda_{{{axn}}}$", "realised factor", None)
    axs[0].set_ylabel("RMSE [m]")
    axs[0].set_yscale("log")
    axs[-1].legend(fontsize=5.4, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("F13  which plant parameter each policy survives", fontsize=8.5,
                 x=0.02, ha="left", y=1.06, color=INK)
    _save(fig, "F13_axis_competence")


def F14_variances(d):
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    _barcmp(ax, d, "policy", [c for c in ("rmse_nominal", "rmse_S2", "seed_std",
                                          "episode_std") if c in d.columns],
            "metric [m]", "F14  conservatism and its variance")
    _save(fig, "F14_variances")


def F15_curriculum(d):
    piv = d.pivot_table(index="trained_through", columns="eval_on", values="rmse")
    order = [s for s in ("nominal", "mass", "lag", "all") if s in piv.index]
    piv = piv.reindex(index=order, columns=[c for c in order if c in piv.columns])
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    im = ax.imshow(piv.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", SEQ), aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.annotate(f"{v:.3f}", (j, i), ha="center", va="center", fontsize=5.8,
                            color="white" if v > np.nanmedian(piv.values) else INK)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    ax.grid(False)
    _tidy(ax, "F15  sequential curriculum (row = trained through)",
          "evaluated on", "trained through")
    fig.colorbar(im, ax=ax, label="RMSE [m]", fraction=0.046)
    _save(fig, "F15_curriculum")


# --------------------------------------------------------------------------- #
# Notebook 5 -- the adaptive arm
# --------------------------------------------------------------------------- #
def F16_encoder_trade(d):
    # accuracy and latency are different measures -> two panels, not a twin axis
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
    xs = np.arange(len(d))
    axs[0].bar(xs, d.r2_overall, 0.6, color=[C[i % len(C)] for i in range(len(d))],
               edgecolor="white", linewidth=0.5)
    for x, v in zip(xs, d.r2_overall):
        axs[0].annotate(f"{v:.3f}", (x, v), ha="center", va="bottom", fontsize=6,
                        color=INK2, xytext=(0, 1.2), textcoords="offset points")
    axs[0].set_xticks(xs); axs[0].set_xticklabels(d.encoder)
    _tidy(axs[0], "(a) accuracy", None, r"$R^2$ overall")
    axs[1].bar(xs, d.lat_p95_ms, 0.6, color=[C[i % len(C)] for i in range(len(d))],
               edgecolor="white", linewidth=0.5)
    axs[1].axhline(20.0, color=INK2, lw=0.9, ls=(0, (4, 2)))
    axs[1].annotate("20 ms period -- the hard constraint", (0, 20.0), xytext=(1, 2),
                    textcoords="offset points", fontsize=5.6, color=INK2)
    for x, v in zip(xs, d.lat_p95_ms):
        axs[1].annotate(f"{v:.2f}", (x, v), ha="center", va="bottom", fontsize=6,
                        color=INK2, xytext=(0, 1.2), textcoords="offset points")
    axs[1].set_xticks(xs); axs[1].set_xticklabels(d.encoder)
    _tidy(axs[1], "(b) single-window latency", None, "p95 [ms]")
    _save(fig, "F16_encoder_trade")


def F17_channel_r2(d):
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ok = d.r2 > 0
    ax.bar(np.arange(len(d)), d.r2, 0.6,
           color=[C[0] if v else C[7] for v in ok], edgecolor="white", linewidth=0.5)
    ax.axhline(0.0, color=INK2, lw=0.8)
    for x, v in zip(np.arange(len(d)), d.r2):
        ax.annotate(f"{v:.3f}", (x, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6, color=INK2,
                    xytext=(0, 1.4 if v >= 0 else -1.4), textcoords="offset points")
    ax.set_xticks(np.arange(len(d))); ax.set_xticklabels(d.channel)
    _tidy(ax, r"F17  per-channel $R^2$ (below 0 = worse than the mean)", None, r"$R^2$")
    _save(fig, "F17_channel_r2")


def F18_closed_loop(d):
    scens = sorted(d.scen.unique())
    fig, axs = plt.subplots(1, len(scens), figsize=(2.4 * len(scens), 2.6), sharey=True)
    axs = np.atleast_1d(axs)
    ctrls = sorted(d.ctrl.unique())
    for a, sc in zip(axs, scens):
        for j, ct in enumerate(ctrls):
            s = d[(d.scen == sc) & (d.ctrl == ct)].sort_values("level")
            a.plot(s.level, s.rmse, ls=DASH[j % len(DASH)], color=C[j % len(C)],
                   marker=MK[j % len(MK)], label=ct)
        _tidy(a, sc, "disturbance level", None)
    axs[0].set_ylabel("RMSE [m]")
    axs[-1].legend(fontsize=5.4, ncol=1, loc="center left",
                   bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("F18  closed loop by scenario -- the zero-level column is "
                 "conservatism, not rejection", fontsize=8, x=0.02, ha="left",
                 y=1.06, color=INK)
    _save(fig, "F18_closed_loop")


def F19_wrench_timeseries(d):
    ch = [c[5:] for c in d.columns if c.startswith("true_")]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.2), sharex=True)
    for k, c in enumerate(ch):
        a = axs.flat[k]
        a.plot(d[f"true_{c}"], "-", color=INK2, lw=1.2, label="truth")
        a.plot(d[f"pred_{c}"], "--", color=C[0], lw=1.3, label="RDP")
        _tidy(a, c, None, None)
        if k == 0:
            a.legend(fontsize=5.6)
    for a in axs[1]:
        a.set_xlabel("step")
    axs[0, 0].set_ylabel("[N]")
    axs[1, 0].set_ylabel(r"[N$\cdot$m]")
    fig.suptitle("F19  RDP prediction against the truth", fontsize=8.5, x=0.02,
                 ha="left", y=1.02, color=INK)
    _save(fig, "F19_wrench_timeseries")


def F20_training_gate(d):
    # reward and saturation are different measures -> two panels
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for j, pol in enumerate(sorted(d.policy.unique())):
        s = d[d.policy == pol]
        axs[0].plot(s.iter, s.reward, color=C[j % len(C)], ls=DASH[j % len(DASH)],
                    label=pol)
        axs[1].plot(s.iter, s.sat, color=C[j % len(C)], ls=DASH[j % len(DASH)],
                    label=pol)
    _tidy(axs[0], "(a) reward", "iteration", "mean reward")
    axs[0].legend(fontsize=5.8)
    axs[1].axhline(0.05, color=INK2, lw=0.9, ls=(0, (4, 2)))
    axs[1].annotate("5 % gate", (0, 0.05), xytext=(1, 2), textcoords="offset points",
                    fontsize=5.6, color=INK2)
    _tidy(axs[1], "(b) collective saturation -- the gate", "iteration", "fraction")
    _save(fig, "F20_training_gate")


# --------------------------------------------------------------------------- #
# Notebook 6 -- the comparison
# --------------------------------------------------------------------------- #
def F21_headline(d):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    fig.subplots_adjust(wspace=0.32)
    cols = [c for c in ("rmse_S1", "rmse_S2", "rmse_S3") if c in d.columns]
    _barcmp(axs[0], d, "ctrl", cols, "RMSE [m]", "(a) accuracy by suite")
    xs = np.arange(len(d))
    axs[1].bar(xs, d.ms_p95, 0.6, color=[C[0] if a else C[7] for a in d.admissible],
               edgecolor="white", linewidth=0.5)
    axs[1].axhline(20.0, color=INK2, lw=0.9, ls=(0, (4, 2)))
    axs[1].annotate("20 ms period", (0, 20.0), xytext=(1, 2),
                    textcoords="offset points", fontsize=5.6, color=INK2)
    for x, v in zip(xs, d.ms_p95):
        axs[1].annotate(f"{v:.2f}", (x, v), ha="center", va="bottom", fontsize=5.6,
                        color=INK2, xytext=(0, 1.2), textcoords="offset points")
    axs[1].set_yscale("log")
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels(d.ctrl, rotation=20, ha="right", fontsize=6)
    over = int((~d.admissible.astype(bool)).sum())
    _tidy(axs[1], "(b) p95 latency" + (" (red = over budget)" if over else ""),
          None, "ms")
    _save(fig, "F21_headline")


def F22_by_condition(d):
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    _barcmp(ax, d, "ctrl", [c for c in ("S1", "S2", "S3") if c in d.columns],
            "RMSE [m]", "F22  every controller on every suite")
    _save(fig, "F22_by_condition")


def F23_radar(d):
    lab = d.columns[1:] if d.columns[0] in ("", "Unnamed: 0") else d.columns[1:]
    names = d.iloc[:, 0].astype(str).values
    vals = d.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").values
    ang = np.linspace(0, 2 * np.pi, len(lab), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    fig, ax = plt.subplots(figsize=(3.8, 3.6), subplot_kw=dict(polar=True))
    for i, nm in enumerate(names):
        v = np.concatenate([vals[i], vals[i][:1]])
        ax.plot(ang, v, color=C[i % len(C)], ls=DASH[i % len(DASH)],
                marker=MK[i % len(MK)], ms=2.6, lw=1.2, label=nm)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(lab, fontsize=6)
    ax.set_ylim(0, 1)
    ax.grid(color="#e6e6e2", lw=0.5)
    ax.set_title("F23  normalised profile (1 = best)", fontsize=8.5, color=INK)
    ax.legend(fontsize=5.2, loc="upper right", bbox_to_anchor=(1.28, 1.14))
    _save(fig, "F23_radar")


def F24_saturation_guard(d):
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    _barcmp(ax, d, "ctrl", [c for c in ("S1", "S2", "S3") if c in d.columns],
            "collective saturation [-]",
            "F24  saturation beside RMSE: above 0.4 the path stopped mattering")
    ax.axhline(0.4, color=C[7], lw=0.9, ls=(0, (4, 2)))
    _save(fig, "F24_saturation_guard")


# --------------------------------------------------------------------------- #
# Notebook 7 -- the clips
# --------------------------------------------------------------------------- #
def F25_contact_sheet(d):
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    xs = np.arange(len(d))
    held = d.rmse < 0.5
    ax.bar(xs, d.rmse, 0.6, color=[C[0] if h else C[7] for h in held],
           edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    for x, v in zip(xs, d.rmse):
        ax.annotate(f"{v:.3g}", (x, v), ha="center", va="bottom", fontsize=6,
                    color=INK2, xytext=(0, 1.2), textcoords="offset points")
    ax.axhline(0.5, color=INK2, lw=0.9, ls=(0, (4, 2)))
    ax.annotate("held the path", (0, 0.5), xytext=(1, 2),
                textcoords="offset points", fontsize=5.6, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([t[:22] for t in d.title.astype(str)], rotation=22, ha="right")
    _tidy(ax, "F25  clip RMSE (red = did not hold the path)", None, "RMSE [m]")
    _save(fig, "F25_contact_sheet")


def F26_ground_tracks(d):
    names, seen = [], set()
    for c in d.columns:
        base = c.rsplit("_", 1)[0]
        if base not in seen and f"{base}_x" in d.columns and f"{base}_y" in d.columns:
            seen.add(base); names.append(base)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for j, nm in enumerate(names):
        ax.plot(d[f"{nm}_x"], d[f"{nm}_y"], color=C[j % len(C)],
                ls=DASH[j % len(DASH)], lw=1.3, label=nm[:24])
    ax.set_aspect("equal")
    _tidy(ax, "F26  ground tracks, identical initial state", "x [m]", "y [m]")
    ax.legend(fontsize=5.2)
    _save(fig, "F26_ground_tracks")


REG = {f.__name__.split("_")[0]: f for f in (
    F1_feasibility, F2_lqr_feedforward, F3_horizon, F3b_noise, F4_corner,
    F5_qr_heatmap, F6_lltc_fit, F7_lltc_locality, F8_lltc_spectrum,
    F9_representations, F10_exploration, F11_mpve, F12_interpretability,
    F13_axis_competence, F14_variances, F15_curriculum, F16_encoder_trade,
    F17_channel_r2, F18_closed_loop, F19_wrench_timeseries, F20_training_gate,
    F21_headline, F22_by_condition, F23_radar, F24_saturation_guard,
    F25_contact_sheet, F26_ground_tracks)}
NAME = {f.__name__.split("_")[0]: f.__name__ for f in REG.values()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="e.g. --only F9 F21")
    ap.add_argument("--list", action="store_true", help="what is buildable now")
    a = ap.parse_args()
    keys = list(REG) if not a.only else [k for k in REG if k in set(a.only)]
    if a.only and not keys:
        raise SystemExit(f"no such figure(s): {a.only}; known: {sorted(REG)}")
    if a.list:
        for k in keys:
            print(f"  {k:5s} {NAME[k]:24s} "
                  f"{'CSV present' if _csv(NAME[k]) is not None else 'csv MISSING'}")
        return
    built, skipped, failed = [], [], []
    for k in keys:
        nm = NAME[k]
        d = _csv(nm)
        if d is None or len(d) == 0:
            skipped.append(nm)
            continue
        try:
            REG[k](d)
            built.append(nm)
        except Exception as e:                       # a bad figure must not stop the set
            failed.append((nm, f"{type(e).__name__}: {e}"))
            plt.close("all")
    print(f"\n  figures dir: {FIG}")
    print(f"  rebuilt {len(built)}/{len(keys)}")
    for nm in skipped:
        print(f"    SKIP   {nm}  (companion csv absent -- report.tex renders a "
              f"labelled placeholder; nothing is invented)")
    for nm, why in failed:
        print(f"    FAILED {nm}  {why}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
