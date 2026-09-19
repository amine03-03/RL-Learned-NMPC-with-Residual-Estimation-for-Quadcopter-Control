#!/usr/bin/env python3
"""Write measured numbers from ``artifacts/`` into the report's results tables.

The report ships with every results cell set to ``\\BL`` --- a grey rule marking
a value awaiting a campaign. This script replaces those cells with the numbers
the notebooks actually produced, in place, so that ``report.tex`` carries
literal figures rather than placeholders.

How it works
------------
Each results table is located by its ``\\label``, and the block between its
first ``\\midrule`` and its ``\\bottomrule`` --- the table body --- is
regenerated from the CSV named in that table's caption. The table's own
structure is the delimiter, so nothing has to be marked up by hand, and
regenerating the whole body makes the script **idempotent**: running it twice
gives the same file, and running it after a fresh campaign updates every
number.

A cell whose CSV has no column for it stays ``\\BL``. That is deliberate. The
alternative --- inventing a plausible number, or silently dropping the column
--- is the failure mode the placeholders exist to prevent.

Safety
------
* ``artifacts/`` is opened **read-only**. Nothing under it is written, moved or
  deleted, so a long campaign's outputs cannot be damaged by running this.
* Only ``report/report.tex`` is written, and a copy of the previous version is
  kept as ``report/report.tex.bak``.
* The script refuses to run when ``report.tex`` has uncommitted changes, unless
  ``--force`` is given, so ``git checkout`` is always a way back.

Usage
-----
    python study/fill_report.py --check     # report coverage, write nothing
    python study/fill_report.py             # fill the tables
    python study/fill_report.py --revert    # restore from the .bak
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEX = os.path.join(ROOT, "report", "report.tex")
ART = os.path.join(ROOT, "artifacts")
BL = r"\BL"


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def load(rel):
    """Read an artefact CSV as a list of dicts, or None when it is absent."""
    path = os.path.join(ART, rel)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows or None


def num(row, key):
    """A float from a CSV cell, or None when absent, empty or non-numeric."""
    if row is None or key not in row:
        return None
    v = (row[key] or "").strip()
    if v == "":
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return None if math.isnan(f) else f


def pick(rows, **kw):
    """The first row matching every key=value pair, or None."""
    if rows is None:
        return None
    for r in rows:
        if all((r.get(k) or "").strip() == str(v) for k, v in kw.items()):
            return r
    return None


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def f(v, nd=4):
    """A fixed-point cell, or \\BL when the value is unavailable."""
    return BL if v is None else f"{v:.{nd}f}"


def fi(v):
    """An integer cell, or \\BL."""
    return BL if v is None else f"{int(round(v)):d}"


def fg(v, nd=3):
    """A general-format cell that stays readable across decades."""
    if v is None:
        return BL
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
        m, e = f"{v:.{nd}e}".split("e")
        return rf"${m}\times10^{{{int(e)}}}$"
    return f"{v:.{nd}f}"


def fpct(v, nd=1):
    return BL if v is None else rf"{v * 100:.{nd}f}\%"


def yn(row, key):
    if row is None or key not in row:
        return BL
    return "yes" if (row[key] or "").strip().lower() in ("true", "1", "yes") else "no"


def slash(*cells):
    """Several controllers sharing one cell, as the table's header promises."""
    return " / ".join(cells)


# --------------------------------------------------------------------------- #
# per-table generators
#
# Each returns the complete table body: every row between \midrule and
# \bottomrule, including any internal \midrule. Returning None means the source
# is missing, and the table is left exactly as it is.
# --------------------------------------------------------------------------- #
GEN = {}


def table(label, source):
    def deco(fn):
        GEN[label] = (source, fn)
        return fn
    return deco


@table("tab:res-classic", "lltc/nb1_classical.csv")
def _classic(rows):
    out = []
    for path in ("hover", "circle", "fig8", "square"):
        for ctrl, name in (("LQR", "LQR"), ("LQR (no feed-fwd)", "LQR (no feed-fwd)")):
            r = pick(rows, path=path, ctrl=ctrl)
            out.append(f"{path:6s} & {name:19s} & " + " & ".join([
                f(num(r, "rmse")), f(num(r, "rmse_iqr")), f(num(r, "maxerr")),
                f(num(r, "effort")), fg(num(r, "smooth")), f(num(r, "sat"), 3),
            ]) + r" \\")
    return out


@table("tab:res-horizon", "common/nb1_horizon.csv")
def _horizon(rows):
    out = []
    for N in ("1", "2", "3", "5", "10", "20"):
        r = pick(rows, N=N)
        out.append(f"{N:2s} & " + " & ".join([
            f(num(r, "circle")), f(num(r, "fig8")), f(num(r, "square")),
            f(num(r, "circle_frozen")), f(num(r, "fig8_frozen")),
            f(num(r, "square_frozen")), f(num(r, "ms_p95"), 2),
        ]) + r" \\")
    return out


@table("tab:res-noise", "common/nb1_noise.csv")
def _noise(rows):
    """One row per (noise level, controller). Sharing a row between three
    controllers, as the placeholder layout did, makes every numeric cell three
    values wide and overruns the text block."""
    ctrls = (("LQR", "LQR"), ("NMPC N=1", r"NMPC $N{=}1$"),
             ("NMPC N=10", r"NMPC $N{=}10$"))
    out = []
    for i, lvl in enumerate(("off", "low", "high")):
        if i:
            out.append(r"\addlinespace[2pt]")
        for j, (key, label) in enumerate(ctrls):
            r = pick(rows, noise=lvl, ctrl=key)
            out.append(f"{lvl if j == 0 else '':5s}& {label} & " + " & ".join([
                f(num(r, "rmse")), f(num(r, "maxerr"), 3),
                fg(num(r, "smooth")), f(num(r, "sat"), 3),
            ]) + r" \\")
    deg = []
    for key, _ in ctrls:
        hi = num(pick(rows, noise="high", ctrl=key), "rmse")
        off = num(pick(rows, noise="off", ctrl=key), "rmse")
        deg.append(BL if (hi is None or not off) else f"{hi / off:.2f}")
    out.append(r"\midrule")
    out.append(r"\multicolumn{2}{@{}l}{\textbf{Degradation high/off}} & "
               + slash(*deg) + r" & --- & --- & --- \\")
    return out


@table("tab:res-qr", "common/nb1_qr_spread.csv")
def _qr(rows):
    out = []
    for case, name in (("best", "Best admissible"), ("median", "Median admissible"),
                       ("worst", "Worst admissible"), ("hand", "Hand-chosen operating point")):
        r = pick(rows, case=case)
        # the spread CSV carries RMSE only; effort, smoothness and saturation
        # are not among its columns, so those cells stay marked
        out.append(f"{name} & {f(num(r, 'rmse'))} & {BL} & {BL} & {BL}" + r" \\")
    b, w = num(pick(rows, case="best"), "rmse"), num(pick(rows, case="worst"), "rmse")
    ratio = BL if (b is None or w is None or not b) else f"{w / b:.1f}"
    out.append(r"\midrule")
    out.append(rf"\textbf{{Best-to-worst ratio}} & \textbf{{{ratio}$\times$}} & {BL} & {BL} & --- \\")
    return out


@table("tab:res-cost-spread", "sensitivity/cost_hand_spread.csv")
def _cost_spread(rows):
    out = []
    for path in ("circle", "fig8", "square"):
        rs = [pick(rows, path=path, ctrl=c) for c in ("LQR", "NMPC N=1")]
        rat = [num(r, "ratio") for r in rs]
        out.append(f"{path:6s} & LQR / NMPC $N{{=}}1$ & " + " & ".join([
            slash(*[f(num(r, "best")) for r in rs]),
            slash(*[f(num(r, "worst")) for r in rs]),
            slash(*[BL if v is None else f"{v:.1f}" for v in rat]),
        ]) + r"$\times$ \\")
    return out


@table("tab:res-rep", "acmpc/nb3_representation.csv")
def _rep(rows):
    out = []
    best = {}
    for task in ("circle", "fig8", "stabilize"):
        cells, vals = [], {}
        for rep in ("diag", "chol", "full"):
            v = num(pick(rows, rep=rep, task=task), "rmse")
            vals[rep] = v
            cells.append(f(v))
        ok = {k: v for k, v in vals.items() if v is not None}
        win = min(ok, key=ok.get) if ok else None
        best[task] = ok
        out.append(f"{task} & " + " & ".join(cells) + " & "
                   + (rf"\texttt{{{win}}}" if win else BL) + r" \\")
    spread = []
    for task in ("circle", "fig8", "stabilize"):
        ok = best[task]
        spread.append(BL if len(ok) < 2 else f"{max(ok.values()) - min(ok.values()):.4f}")
    out.append(r"\midrule")
    out.append(r"\textbf{Margin over seed spread} & " + " & ".join(spread) + r" & --- \\")
    return out


@table("tab:res-trainhealth", "acmpc_adaptive/nb5_gate.csv")
def _trainhealth(rows):
    out = []
    for pol in ("Base", "Robust", "Oracle"):
        r = pick(rows, policy=pol)
        out.append(f"{pol} & " + " & ".join([
            f(num(r, "final_reward")), fpct(num(r, "train_sat"), 2),
            fpct(num(r, "crash_rate"), 2), yn(r, "passed"),
        ]) + r" \\")
    return out


@table("tab:res-rdp", "acmpc_adaptive/nb5_estimators.csv")
def _rdp(rows):
    names = (("GRU", r"GRU~\cite{cho2014gru}"), ("LSTM", r"LSTM~\cite{hochreiter1997lstm}"),
             ("TCN", r"TCN~\cite{bai2018tcn}"), ("CNN", "CNN"))
    out, adm = [], []
    for key, label in names:
        r = pick(rows, encoder=key)
        if r is not None and (r.get("admissible", "").lower() == "true"):
            adm.append((key, num(r, "r2_overall")))
        out.append(f"{label} & " + " & ".join([
            f(num(r, "r2_overall"), 4), f(num(r, "r2_force"), 4), f(num(r, "r2_moment"), 4),
            fi(num(r, "params")), f(num(r, "lat_p95_ms"), 2), yn(r, "admissible"),
        ]) + r" \\")
    ship = max(adm, key=lambda t: (t[1] is not None, t[1]))[0] if adm else None
    out.append(r"\midrule")
    out.append(r"\textbf{Shipped} & \multicolumn{6}{l}{"
               + (ship or BL)
               + r"\ \ (chosen among the admissible encoders by $R^2$)} \\")
    return out


@table("tab:res-scenarios", "acmpc_adaptive/nb5_scenarios.csv")
def _scenarios(rows):
    cols = ("Base", "Robust", "Oracle", "RDP-A", "RDP-B", "RDP-C")
    groups = (("central", "central payload (fraction of weight)",
               (("0.0", "0.000"), ("0.03", "0.030"), ("0.06", "0.060"), ("0.1", "0.100"))),
              ("asym", "arm-tip payload, offset CG",
               (("0.0", "0.000"), ("0.004", "0.004"), ("0.028", "0.028"), ("0.044", "0.044"))),
              ("slung", r"slung load (excitation period, \si{\second})",
               (("0.0", "0.0"), ("1.2", "1.2"), ("2.0", "2.0"), ("4.0", "4.0"), ("6.0", "6.0"))))
    out = []
    for i, (scen, head, levels) in enumerate(groups):
        if i:
            out.append(r"\midrule")
        out.append(r"\multicolumn{7}{@{}l}{\itshape " + head + r"} \\")
        for key, shown in levels:
            r = pick(rows, scen=scen, level=key)
            out.append(rf"\quad {shown} & "
                       + " & ".join(f(num(r, c)) for c in cols) + r" \\")
    return out


@table("tab:ledger", "common/ledger.csv")
def _ledger(rows):
    names = (("LQR", "LQR + feed-forward", "13"),
             ("NMPC N=1", r"NMPC $N{=}1$", "13"),
             ("NMPC N=10", r"NMPC $N{=}10$", "13"),
             ("LLTC N=1", r"\LLTC{} $N{=}1$", "13"),
             ("AC-MPC N=1", r"\ACMPC{} $N{=}1$", r"\textbf{0}"),
             ("Adaptive AC-MPC N=1", r"Adaptive \ACMPC{} $N{=}1$", r"\textbf{0}"))
    out = []
    for key, label, tuned in names:
        r = pick(rows, ctrl=key)
        out.append(f"{label} & " + " & ".join([
            f(num(r, "rmse_S1")), f(num(r, "rmse_S2")), f(num(r, "rmse_S3")),
            f(num(r, "ms_p95"), 2), yn(r, "admissible"),
        ]) + f" & {tuned}" + r" \\")
    return out


@table("tab:res-degradation", "common/nb6_degradation.csv")
def _degradation(rows):
    names = (("LQR", "LQR + feed-forward"), ("NMPC N=1", r"NMPC $N{=}1$"),
             ("NMPC N=10", r"NMPC $N{=}10$"), ("LLTC N=1", r"\LLTC{} $N{=}1$"),
             ("AC-MPC N=1", r"\ACMPC{} $N{=}1$"),
             ("Adaptive AC-MPC N=1", r"Adaptive \ACMPC{} $N{=}1$"))
    out = []
    for key, label in names:
        r = pick(rows, ctrl=key)
        r1, r3, ch = num(r, "rank_S1"), num(r, "rank_S3"), num(r, "rank_change")
        rank = BL if (r1 is None or r3 is None) else rf"{int(r1)} $\to$ {int(r3)}"
        chg = BL if ch is None else (rf"$-{abs(int(ch))}$" if ch < 0 else f"$+{int(ch)}$" if ch else "0")
        out.append(f"{label} & " + " & ".join([
            f(num(r, "S2/S1"), 2) + r"$\times$", f(num(r, "S3/S1"), 2) + r"$\times$",
            rank, chg,
        ]) + r" \\")
    return out


@table("tab:res-mismatch-hand", "sensitivity/model_hand_breakpoints.csv")
def _mismatch_hand(rows):
    out = []
    for ax, label in (("m", r"$\lambda_m$ (mass)   "), ("J", r"$\lambda_J$ (inertia)")):
        r = pick(rows, axis=ax)
        bl, bn = num(r, "break_LQR"), num(r, "break_NMPC")
        above, below = num(r, "NMPC_loses_above"), num(r, "NMPC_loses_below")
        out.append(f"{label} & " + " & ".join([
            (BL if bl is None else f"{bl:.2f}") + r"$\times$",
            (BL if bn is None else f"{bn:.2f}") + r"$\times$",
            BL if above is None else f"{above:.2f}",
            BL if below is None else f"{below:.2f}",
        ]) + r" \\")
    return out


@table("tab:res-mismatch-learned", "sensitivity/model_learned_breakpoints.csv")
def _mismatch_learned(rows):
    out = []
    for ax, label in (("m", r"$\lambda_m$ (mass)   "), ("J", r"$\lambda_J$ (inertia)")):
        r = pick(rows, axis=ax)
        a, b = num(r, "AC-MPC N=1"), num(r, "Adaptive AC-MPC N=1")
        out.append(f"{label} & " + " & ".join([
            (BL if a is None else f"{a:.2f}") + r"$\times$",
            (BL if b is None else f"{b:.2f}") + r"$\times$",
            BL, BL,
        ]) + r" \\")
    return out


@table("tab:res-rdp-window", "sensitivity/rdp_window.csv")
def _rdp_window(rows):
    enc = ("GRU", "LSTM", "TCN", "CNN")
    warm = {"16": r"\SI{320}{\milli\second}", "32": r"\SI{640}{\milli\second}",
            "64": r"\SI{1.28}{\second}", "128": r"\SI{2.56}{\second}"}
    out = [r"\multicolumn{6}{@{}l}{\itshape overall $R^2$} \\"]
    cells = {}
    for H in ("16", "32", "64", "128"):
        vals = [num(pick(rows, H=H, encoder=e), "r2_overall") for e in enc]
        cells[H] = vals
        out.append(rf"\quad {H:3s} & {warm[H]:24s} & "
                   + " & ".join(f(v, 4) for v in vals) + r" \\")
    out.append(r"\midrule")
    out.append(r"\multicolumn{6}{@{}l}{\itshape p95 inference latency, one window "
               r"[\si{\milli\second}]} \\")
    for H in ("16", "128"):
        vals = [num(pick(rows, H=H, encoder=e), "lat_p95_ms") for e in enc]
        out.append(rf"\quad {H:3s} & ---                      & "
                   + " & ".join(f(v, 2) for v in vals) + r" \\")
    flat = [(H, v) for H in cells for v in cells[H] if v is not None]
    short = BL
    if flat:
        best = max(v for _, v in flat)
        ok = [int(H) for H, v in flat if v >= 0.95 * best]
        short = rf"$H={min(ok)}$"
    out.append(r"\midrule")
    out.append(r"\multicolumn{2}{@{}l}{Shortest window within \SI{5}{\percent} of the best cell}"
               "\n\t\t\t& " + rf"\multicolumn{{4}}{{r}}{{{short}}} \\")
    return out


@table("tab:res-seeds", "domrand/seed_study.csv")
def _seeds(rows):
    ac = load("acmpc/nb3_seeds.csv")
    out = []
    for pol, label in (("nominal", "Nominal-only"), ("DR-all", "DR, all axes")):
        rs = [r for r in (rows or []) if (r.get("policy") or "") == pol]
        vals = [num(r, "rmse_nominal") for r in rs]
        vals = [v for v in vals if v is not None]
        eps = [num(r, "episode_std") for r in rs]
        eps = [v for v in eps if v is not None]
        out.append(f"{label} & "
                   + (BL if len(vals) < 2 else f"{max(vals) - min(vals):.4f}")
                   + " & " + (BL if not eps else f"{sum(eps) / len(eps):.4f}") + r" \\")
    av = [num(r, "rmse") for r in (ac or [])]
    av = [v for v in av if v is not None]
    out.append(r"\ACMPC{} (three seeds) & "
               + (BL if len(av) < 2 else f"{max(av) - min(av):.4f}")
               + f" & {BL}" + r" \\")
    pool = ([max(vals) - min(vals)] if len(vals) >= 2 else []) + \
           ([max(av) - min(av)] if len(av) >= 2 else [])
    floor = BL if not pool else f"{max(pool):.4f}"
    out.append(r"\midrule")
    out.append(r"\textbf{Seed spread used as the comparison floor} & \textbf{"
               + floor + r"\,\si{\metre}} & --- \\")
    return out


@table("tab:res-dr", "domrand/T6_robustness.csv")
def _dr(rows):
    """T6_robustness.csv is long-format: one metric name per row."""
    def metric(name):
        for r in rows:
            if (r.get("metric") or "").strip() == name:
                return num(r, "value")
        return None

    nom_n = metric("RMSE on the nominal plant, nominal-only policy")
    dr_n = metric("RMSE on the nominal plant, DR-all policy")
    nom_2 = metric("RMSE on S2, nominal-only policy")
    dr_2 = metric("RMSE on S2, DR-all policy")
    prem, gain = metric("conservatism premium [%] (signed)"), metric("robustness gain on S2 [%]")

    out = [f"Nominal-only & {f(nom_n)} & {f(nom_2)} & {BL} & {BL}" + r" \\",
           f"DR, all axes & {f(dr_n)} & {f(dr_2)} & {BL} & {BL}" + r" \\",
           # this suite reports the two headline policies only
           f"DR + measurement noise & {BL} & {BL} & {BL} & {BL}" + r" \\",
           f"NMPC $N{{=}}1$ (no learning) & {BL} & {BL} & {BL} & {BL}" + r" \\",
           r"\midrule",
           r"\textbf{Conservatism premium (signed)} & \textbf{"
           + (BL if prem is None else f"{prem:.2f}") + r"\,\%} & --- & --- & " + BL + r"\,\% \\",
           r"\textbf{Robustness gain on S2 (signed)} & --- & \textbf{"
           + (BL if gain is None else f"{gain:.2f}") + r"\,\%} & --- & --- \\"]
    return out


@table("tab:res-lltc", "lltc/nb2_horizon_equivalence.csv")
def _lltc(rows):
    """The CSV is indexed by path with one column per controller; the table is
    the transpose. Effort and solve time are not among its columns."""
    ctrls = (("NMPC N=1", r"NMPC $N{=}1$"), ("NMPC N=3", r"NMPC $N{=}3$"),
             ("NMPC N=10", r"NMPC $N{=}10$"), ("LLTC N=1", r"\LLTC{} $N{=}1$"))
    out = []
    for key, label in ctrls:
        cells = [f(num(pick(rows, path=pth), key)) for pth in ("circle", "fig8", "square")]
        out.append(f"{label} & " + " & ".join(cells) + f" & {BL} & {BL}" + r" \\")
    rat = [num(pick(rows, path=pth), "ratio LLTC/NMPC N=1") for pth in ("circle", "fig8", "square")]
    out.append(r"\midrule")
    out.append(r"\textbf{Ratio \LLTC/NMPC $N{=}1$} & "
               + " & ".join(BL if v is None else f"{v:.1f}$\\times$" for v in rat)
               + r" & --- & --- \\")
    return out


@table("tab:res-lltcfit", "lltc/nb2_fit.csv")
def _lltcfit(rows):
    """The fit artefact records the shipped configuration only; the three
    ablation rows are left marked."""
    r = rows[0] if rows else None
    out = ["Baseline & " + " & ".join([
        f(num(r, "R2"), 6), fg(num(r, "NRMSE")), fpct(num(r, "acceptance"), 0),
        fi(num(r, "n")), f(num(r, "reach_m"), 4)]) + r" \\"]
    for name in ("Reduced terminal weighting", "Matched displacement",
                 "Increased candidate budget"):
        out.append(f"{name} & " + " & ".join([BL] * 5) + r" \\")
    return out


@table("tab:res-gaps", "acmpc_adaptive/nb5_scenarios.csv")
def _gaps(rows):
    """Both gaps are differences of columns already in the scenario sweep,
    taken as medians over the levels of each scenario, as the caption states."""
    out = []
    for scen in ("central", "asym", "slung"):
        rs = [r for r in rows if (r.get("scen") or "") == scen]
        voi, coe = [], []
        for r in rs:
            o = num(r, "Oracle")
            rb = num(r, "Robust")
            rdp = [v for v in (num(r, c) for c in ("RDP-A", "RDP-B", "RDP-C")) if v is not None]
            if o is not None and rb is not None:
                voi.append(rb - o)
            if o is not None and rdp:
                coe.append(min(rdp) - o)
        # the caption says medians over levels, so take the median
        def m(xs):
            if not xs:
                return BL
            q = sorted(xs)
            n = len(q)
            med = q[n // 2] if n % 2 else 0.5 * (q[n // 2 - 1] + q[n // 2])
            return f"{med:+.4f}"
        out.append(f"{scen} & {m(voi)} & {m(coe)}" + r" \\")
    return out


# --------------------------------------------------------------------------- #
# splicing
# --------------------------------------------------------------------------- #
TABLE_RE = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.S)


def splice(src, label, body_lines):
    """Replace one table's body, keeping the surrounding indentation."""
    for m in TABLE_RE.finditer(src):
        blk = m.group(0)
        lab = re.search(r"\\label\{([^}]+)\}", blk)
        if not lab or lab.group(1) != label:
            continue
        if "\\midrule" not in blk or "\\bottomrule" not in blk:
            return src, "no \\midrule/\\bottomrule"
        head, rest = blk.split("\\midrule", 1)
        _, tail = rest.split("\\bottomrule", 1)
        indent = re.search(r"\n([\t ]*)\\bottomrule", blk)
        pad = indent.group(1) if indent else "\t\t\t"
        body = "\n" + "\n".join(pad + ln for ln in body_lines) + "\n" + pad
        new = head + "\\midrule" + body + "\\bottomrule" + tail
        return src[:m.start()] + new + src[m.end():], None
    return src, "label not found"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report coverage, write nothing")
    ap.add_argument("--force", action="store_true", help="run despite uncommitted changes")
    ap.add_argument("--revert", action="store_true", help="restore report.tex from the .bak")
    a = ap.parse_args()

    bak = TEX + ".bak"
    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy2(bak, TEX)
        print(f"restored {TEX} from {bak}")
        return 0

    if not a.check and not a.force:
        try:
            dirty = subprocess.run(["git", "status", "--porcelain", "--", TEX],
                                   cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            dirty = ""
        if dirty:
            sys.exit("report.tex has uncommitted changes.\n"
                     "Commit them, or pass --force (a .bak is kept either way).")

    src = open(TEX, encoding="utf-8").read()
    before = src.count(BL)
    filled, skipped = [], []

    for label, (source, fn) in GEN.items():
        rows = load(source)
        if rows is None:
            skipped.append((label, f"missing artifacts/{source}"))
            continue
        body = fn(rows)
        if body is None:
            skipped.append((label, "generator declined"))
            continue
        left = sum(ln.count(BL) for ln in body)
        src, err = splice(src, label, body)
        if err:
            skipped.append((label, err))
        else:
            filled.append((label, source, left))

    after = src.count(BL)
    print(f"{'table':28s} {'source':42s} cells left")
    print("-" * 84)
    for label, source, left in sorted(filled):
        print(f"{label:28s} {source:42s} {left:>4d}")
    if skipped:
        print("\nnot filled:")
        for label, why in sorted(skipped):
            print(f"  {label:28s} {why}")
    print(f"\n\\BL cells: {before} -> {after}  ({before - after} replaced)")

    if a.check:
        print("\n--check: nothing written.")
        return 0

    shutil.copy2(TEX, bak)
    open(TEX, "w", encoding="utf-8").write(src)
    print(f"\nwrote {TEX}\nbackup {bak}")
    print("artifacts/ was opened read-only; nothing under it was modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
