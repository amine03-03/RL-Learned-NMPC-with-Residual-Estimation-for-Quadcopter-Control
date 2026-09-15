"""study_prelude.py -- scale table, evaluation helpers and notebook furniture.
Implements §7.1.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import x500_core_jax as X

#: Scale table selected by ``X500_SCALE``.  ``smoke`` exists **only** to verify
#: that the pipeline executes.
SCALES = {
    "smoke": dict(n_env=16, T_rollout=16, minib=8, epochs=4, iters=6, iters_sweep=4,
                  hid=64, ilqr=5, n_eval=8, T_eval=250, est_epochs=2, est_batch=128),
    "full": dict(n_env=512, T_rollout=64, minib=32, epochs=10, iters=400,
                 iters_sweep=150, hid=1024, ilqr=25, n_eval=256, T_eval=1200,
                 est_epochs=60, est_batch=4096),
}
#: An intermediate table.  Not in §7.1; added because `smoke` produces
#: undertrained policies by design and `full` costs ~10 h, and a build has to be
#: verifiable in between.  Its numbers are **still not results** -- only `full`
#: rows may go into a write-up (§13).
SCALES["medium"] = dict(n_env=128, T_rollout=32, minib=16, epochs=6, iters=80,
                        iters_sweep=30, hid=256, ilqr=10, n_eval=64, T_eval=600,
                        est_epochs=15, est_batch=1024)

SCALE = os.environ.get("X500_SCALE", "smoke")
if SCALE not in SCALES:
    raise SystemExit(f"X500_SCALE={SCALE!r} unknown; choose from {sorted(SCALES)}")
CFG = dict(SCALES[SCALE])
CFG["scale"] = SCALE
CFG["disp"] = 0.35          # LLTC candidate displacement scale (§8.2)
CFG["n_cand"] = 4096 if SCALE == "full" else 512

DR_BUDGET = os.environ.get("X500_DR_BUDGET", "reduced")     # §8.4
DR_BUDGETS = {
    "full": dict(iters=CFG["iters"], seq_iters=120, seeds=3, bins=8,
                 eval_ep=CFG["n_eval"], eval_T=CFG["T_eval"]),
    "reduced": dict(iters=200, seq_iters=60, seeds=2, bins=4, eval_ep=16, eval_T=200),
}
if SCALE == "smoke":
    DR_BUDGETS["full"] = dict(iters=CFG["iters"], seq_iters=4, seeds=2, bins=3,
                              eval_ep=8, eval_T=120)
    DR_BUDGETS["reduced"] = dict(iters=CFG["iters"], seq_iters=3, seeds=2, bins=3,
                                 eval_ep=8, eval_T=100)


def header(name):
    """Scale banner.  Printed by every notebook's cell 1."""
    X.banner(f"{name}   |   X500_SCALE={SCALE}   |   DR budget={DR_BUDGET}")
    print(f"  n_env={CFG['n_env']}  iters={CFG['iters']}  hid={CFG['hid']}  "
          f"ilqr={CFG['ilqr']}  n_eval={CFG['n_eval']}  T_eval={CFG['T_eval']}")
    if SCALE == "smoke":
        print("  !! smoke scale: learned-controller numbers below are NOT results.")
        print("     Expect the signature -- hand-built controllers (LQR, NMPC)")
        print("     unaffected, every learned controller diverging.")
    print()


def section(i, name, purpose, consumes="", produces=""):
    X.banner(f"[{i}] {name}", "-")
    print(f"  purpose : {purpose}")
    if consumes:
        print(f"  consumes: {consumes}")
    if produces:
        print(f"  produces: {produces}")
    print()


def table(df, title, note="", csv=None, float_fmt="%.4f"):
    """Print a result table and write its machine-readable companion."""
    print(f"\n=== {title} ===")
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        print(df.to_string(index=False, float_format=lambda v: float_fmt % v))
    if note:
        print(f"  note: {note}")
    if csv:
        p = X.apath(*csv) if isinstance(csv, (list, tuple)) else csv
        df.to_csv(p, index=False)
        print(f"  -> {os.path.relpath(p, X.ARTIFACTS)}")
    return df


def checkpoint(**paths):
    print("  checkpoints:")
    for k, v in paths.items():
        p = X.apath(*v) if isinstance(v, (list, tuple)) else v
        ok = "present" if os.path.exists(p) else "MISSING"
        print(f"    {k:22s} {os.path.relpath(p, X.ARTIFACTS):40s} {ok}")


def nominal_spec(**kw):
    return X.nominal_spec(**kw)


def disturbed_spec(**kw):
    return X.disturbed_spec(**kw)


def ood_spec(**kw):
    return X.ood_spec(**kw)


def ev_env(kind, n=None, spec=None, seed=0, ep_len=None, **kw):
    """An evaluation environment: fixed seed, long episodes, no reward shaping."""
    n = CFG["n_eval"] if n is None else n
    ep_len = (CFG["T_eval"] + 1000) if ep_len is None else ep_len
    paths = (kind,) if isinstance(kind, str) else tuple(kind)
    return X.Env(n, seed, ep_len, spec or nominal_spec(), paths, **kw)


def controllers(env=None, which=("LQR", "NMPC1", "NMPC10"), n_iter=None):
    """The hand-built controller set.  **PID is deliberately absent** (§5.12)."""
    n_iter = CFG["ilqr"] if n_iter is None else n_iter
    out = {}
    if "LQR" in which:
        out["LQR"] = X.make_lqr_ctrl()
    if "LQR-noff" in which:
        out["LQR (no feed-fwd)"] = X.make_lqr_ctrl(feedforward=False)
    for w in which:
        if w.startswith("NMPC") and w[4:].isdigit():
            N = int(w[4:])
            out[f"NMPC N={N}"] = X.make_nmpc_ctrl(env, N=N, n_iter=n_iter)
    return out


def scale_guard(df, ctrl_col="ctrl", rmse_col="rmse", thresh=0.5, verbose=True):
    """T-20: detect the undertrained-policy signature and warn naming the table.

    The signature is a *split along the training axis*: hand-built controllers
    (LQR, NMPC) unaffected while every learned controller diverges.  That is a
    statement about the scale table, not about a controller class, and saying so
    is the difference between a negative result and a wrong one.
    """
    HAND = ("LQR", "NMPC", "PID")
    is_hand = df[ctrl_col].astype(str).str.startswith(HAND)
    hand, learned = df[is_hand], df[~is_hand]
    if len(hand) == 0 or len(learned) == 0:
        return False
    bad = (learned[rmse_col] > thresh).mean()
    fine = (hand[rmse_col] <= thresh).mean()
    fired = bool(bad >= 0.5 and fine >= 0.5)
    if fired and verbose:
        print("!" * 78)
        print(f"!! UNDERTRAINED-POLICY SIGNATURE: {100*bad:.0f}% of learned rows "
              f"exceed {thresh} m RMSE while {100*fine:.0f}% of hand-built rows do not.")
        print(f"!! The split falls along the training axis, so this is a property "
              f"of scale table X500_SCALE={SCALE!r}, not of the controller class.")
        print(f"!! Re-run at X500_SCALE=full before reading these rows as results.")
        print("!" * 78)
    return fired


class Timer:
    """§8.4: instrument before optimising.  Do not claim a speed-up you did not
    measure."""

    def __init__(self):
        self.rows = []

    def __call__(self, label, **meta):
        return _TimerCtx(self, label, meta)

    def df(self):
        return pd.DataFrame(self.rows)


class _TimerCtx:
    def __init__(self, parent, label, meta):
        self.p, self.label, self.meta = parent, label, meta

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.p.rows.append(dict(label=self.label,
                                seconds=time.perf_counter() - self.t0, **self.meta))
        return False
