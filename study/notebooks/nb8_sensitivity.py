# %% [markdown]
# # Notebook 8 — sensitivity: what each controller is actually fragile to
#
# **Establishes.** How far each controller survives (a) a plant that is not the
# one it plans with, and (b) a stage cost that was chosen badly. The hand-built
# arms are stressed on both axes; the learned arms on the plant axis, because
# they have no hand-chosen cost to stress. The adaptive arm additionally gets a
# horizon sweep, which needs a *retrain per horizon*: the cost-map head emits
# `N × REP_DIM` numbers, so an actor trained at `N=1` cannot be evaluated at
# `N=3`.
#
# **Every sweep runs on `circle`, `fig8` and `square`,** because the three differ
# in how close the reference sits to the feasibility cap (NB1: `square` is capped
# on 100 % of radii, `fig8` on 6.2 %, `circle` on none) and a controller's
# fragility is not the same on a path that already has the collective near its
# box.
#
# **Produces.** `sensitivity/*.csv`, figures F27–F31.
#
# A missing checkpoint **skips its section and says so**; it does not abort and
# nothing is synthesised in its place.

# %%
import _nbinit  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import adaptive_core_jax as A
import x500_core_jax as X
import study_prelude as S
import viz as V

S.header("Notebook 8 - sensitivity and stress")
X.style()
FIG = V.figures_dir()

PATHS = ("circle", "fig8", "square")
NIT = S.CFG["ilqr"]
#: evaluation budget.  Deliberately smaller than the headline suites: this
#: notebook runs hundreds of cells, and a sweep's SHAPE is what is being read,
#: not a single cell to four decimals.
N_EV = max(S.CFG["n_eval"] // 2, 8)
T_EV = max(S.CFG["T_eval"] // 3, 80)
SPEED = (1.2, 1.2)          # pinned: a speed band would blur every sweep

#: Multiplicative plant-parameter stress.  Wide on purpose -- the point is to
#: find where each controller BREAKS, not to confirm it survives S2.  `tau` is
#: one-sided because tau < 1 is a *faster* rotor than nominal, which is not a
#: stress; the interesting direction is lag.
LEVELS = {
    "m":   (0.60, 0.75, 0.90, 1.00, 1.15, 1.35, 1.60),
    "D":   (0.25, 0.50, 1.00, 1.75, 2.50, 3.25, 4.00),
    "tau": (1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 6.00),
    "T":   (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30),
    "Kw":  (0.40, 0.60, 0.80, 1.00, 1.30, 1.55, 1.80),
    "J":   (0.50, 0.70, 0.85, 1.00, 1.30, 1.60, 2.00),
}
AXES = tuple(LEVELS)


def mismatch_env(axis, level, path, seed=4242, n=None, T=None, adapt=False, **kw):
    """An environment whose plant is pinned off-nominal on ONE axis.

    Everything else is held at nominal and the speed band is a point, so the
    only thing moving across a sweep is the parameter named.
    """
    spec = S.nominal_spec(speed=SPEED)
    fixed = {f"lam_{axis}": float(level)}
    n = N_EV if n is None else n
    T = T_EV if T is None else T
    if adapt:
        return A.AdaptEnv(n, seed, 100000, spec, "central", 0.0, paths=(path,),
                          fixed=fixed, **kw)
    return X.Env(n, seed, 100000, spec, (path,), fixed=fixed, **kw)


def ev(env, ctrl, T=None):
    return X.stats(X.rollout_eval(env, ctrl, T or T_EV, warmup=30))


# %% [markdown]
# ## 1 — the hand-built arms against a plant they did not plan with
#
# LQR and NMPC $N{=}1$ share the same model, the same weights and the same
# feed-forward. They differ only in that one solves online. This sweep asks
# whether the online solve buys any *robustness*, as opposed to the accuracy it
# was shown to buy in NB1.

# %%
S.section(1, "model mismatch, hand-built", "LQR and NMPC N=1 off-nominal on six "
          "plant axes x three paths", produces="sensitivity/model_hand.csv, F27")

rows = []
for path in PATHS:
    for axis in AXES:
        for lv in LEVELS[axis]:
            env = mismatch_env(axis, lv, path)
            for name, ctrl in (("LQR", X.make_lqr_ctrl()),
                               ("NMPC N=1", X.make_nmpc_ctrl(None, N=1, n_iter=NIT))):
                st = ev(env, ctrl)
                rows.append(dict(path=path, axis=axis, level=lv, ctrl=name,
                                 rmse=st["rmse"], sat=st["sat"], crash=st["crash"],
                                 bound_frac=st["bound_frac"]))
MH = pd.DataFrame(rows)
MH.to_csv(X.apath("sensitivity", "model_hand.csv"), index=False)

piv = MH.pivot_table(index=["axis", "ctrl"], columns="level", values="rmse")
S.table(piv.reset_index(), "Model mismatch, hand-built (RMSE [m], median over paths)",
        note="the level column is the multiplicative factor on that plant parameter; "
             "1.0 is nominal", csv=("sensitivity", "model_hand_pivot.csv"))


def _breakpoint(df, ctrl, axis, thresh=3.0):
    """Factor at which RMSE first exceeds `thresh` x its nominal value.

    Reported instead of a bare worst case because 'where it breaks' is the
    question; a controller that degrades gracefully to 2x over a 4x parameter
    error is a different object from one that is fine until it is not.
    """
    d = df[(df.ctrl == ctrl) & (df.axis == axis)].groupby("level").rmse.median()
    if 1.0 not in d.index:
        return np.nan
    base = d.loc[1.0]
    bad = d[d > thresh * base]
    return float(bad.index.min()) if len(bad) else np.nan


bp = pd.DataFrame([
    dict(axis=a, **{c: _breakpoint(MH, c, a) for c in ("LQR", "NMPC N=1")})
    for a in AXES])
S.table(bp, "Break factor (first level at 3x the nominal RMSE; NaN = never)",
        note="NaN means the controller never degraded 3x anywhere in the swept "
             "range -- read it beside the range, not alone",
        csv=("sensitivity", "model_hand_breakpoints.csv"))

fig, axs = plt.subplots(len(PATHS), len(AXES), figsize=(2.05 * len(AXES),
                                                        1.95 * len(PATHS)),
                        sharex="col")
for i, path in enumerate(PATHS):
    for j, axis in enumerate(AXES):
        a = axs[i, j]
        for k, c in enumerate(("LQR", "NMPC N=1")):
            d = MH[(MH.path == path) & (MH.axis == axis) & (MH.ctrl == c)]
            d = d.sort_values("level")
            a.plot(d.level, d.rmse, color=["#2a78d6", "#eb6834"][k],
                   ls=["-", "--"][k], marker=["o", "s"][k], ms=3, lw=1.4, label=c)
        a.axvline(1.0, color="#8a8a84", lw=0.7, ls=(0, (3, 2)))
        a.set_yscale("log")
        a.grid(color="#e6e6e2", lw=0.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        if i == 0:
            a.set_title(rf"$\lambda_{{{axis}}}$", fontsize=8)
        if j == 0:
            a.set_ylabel(f"{path}\nRMSE [m]", fontsize=7)
        if i == len(PATHS) - 1:
            a.set_xlabel("factor", fontsize=7)
axs[0, 0].legend(fontsize=5.6)
fig.suptitle("F27  hand-built controllers against plant mismatch "
             "(dashed line = nominal)", fontsize=9, x=0.02, ha="left", y=1.005)
fig.savefig(f"{FIG}/F27_model_mismatch_hand.png", bbox_inches="tight", dpi=200)
MH.to_csv(f"{FIG}/F27_model_mismatch_hand.csv", index=False)
plt.close(fig)

print("\n  ANALYSIS -- where each hand-built arm breaks:")
for a_ in AXES:
    l_, n_ = _breakpoint(MH, "LQR", a_), _breakpoint(MH, "NMPC N=1", a_)
    f = lambda v: "never in range" if not np.isfinite(v) else f"{v:g}x"
    print(f"    lambda_{a_:4s}: LQR {f(l_):>14s}   NMPC N=1 {f(n_):>14s}")


# %% [markdown]
# ## 2 — the hand-built arms against a stage cost chosen badly
#
# NB1 sweeps $(Q_{pos}, R_{rate})$ for NMPC $N{=}1$ on the circle and finds a
# 10.1× best-to-worst ratio. This widens that: **both** hand-built controllers,
# **three** paths, and a grid that runs past the admissible region on purpose.
# This is the quantity §8.3 removes, so it is worth measuring where it is worst
# rather than where it is comfortable.

# %%
S.section(2, "cost mismatch, hand-built", "a deliberately wide (Q_pos, R_rate) "
          "grid for LQR and NMPC N=1 on three paths",
          produces="sensitivity/cost_hand.csv, F28")

QPOS = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
RRATE = (0.02, 0.05, 0.2, 1.0, 5.0, 20.0, 50.0)

rows = []
for path in PATHS:
    for qp in QPOS:
        for rr in RRATE:
            Q = np.diag([qp, qp, qp, 4.0, 4.0, 4.0, 1.0, 1.0, 0.5])
            R = np.diag([0.5, rr, rr, rr])
            env = X.Env(N_EV, 4242, 100000, S.nominal_spec(speed=SPEED), (path,))
            try:
                K, _, Ad, Bd = X.dlqr(Q, R)
                stable = float(np.abs(np.linalg.eigvals(Ad - Bd @ K)).max()) < 1.0
            except Exception:
                K, stable = None, False
            if K is not None:
                st = ev(env, X.make_lqr_ctrl(K=K))
                rows.append(dict(path=path, Q_pos=qp, R_rate=rr, ctrl="LQR",
                                 rmse=st["rmse"], sat=st["sat"], crash=st["crash"],
                                 stabilising=stable))
            env = X.Env(N_EV, 4242, 100000, S.nominal_spec(speed=SPEED), (path,))
            st = ev(env, X.make_nmpc_ctrl(None, N=1, n_iter=NIT, Q=Q, R=R))
            rows.append(dict(path=path, Q_pos=qp, R_rate=rr, ctrl="NMPC N=1",
                             rmse=st["rmse"], sat=st["sat"], crash=st["crash"],
                             stabilising=True))
CH = pd.DataFrame(rows)
CH.to_csv(X.apath("sensitivity", "cost_hand.csv"), index=False)

sp = []
for path in PATHS:
    for c in ("LQR", "NMPC N=1"):
        d = CH[(CH.path == path) & (CH.ctrl == c)].rmse
        sp.append(dict(path=path, ctrl=c, best=float(d.min()), median=float(d.median()),
                       worst=float(d.max()), ratio=float(d.max() / max(d.min(), 1e-9))))
SP = pd.DataFrame(sp)
S.table(SP, "Cost-weight spread over the wide grid",
        note="`ratio` is how much of 'controller performance' is really the "
             "human's choice of weights -- the quantity a learned cost removes",
        csv=("sensitivity", "cost_hand_spread.csv"))

SEQ = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
fig, axs = plt.subplots(2, len(PATHS), figsize=(2.35 * len(PATHS), 4.6))
vmin = float(CH.rmse.min())
vmax = float(np.nanpercentile(CH.rmse, 98))
for i, c in enumerate(("LQR", "NMPC N=1")):
    for j, path in enumerate(PATHS):
        a = axs[i, j]
        p = CH[(CH.ctrl == c) & (CH.path == path)].pivot_table(
            index="Q_pos", columns="R_rate", values="rmse")
        im = a.imshow(p.values, cmap=SEQ, origin="lower", aspect="auto",
                      vmin=vmin, vmax=vmax)
        best = np.unravel_index(np.nanargmin(p.values), p.values.shape)
        a.add_patch(plt.Rectangle((best[1] - .5, best[0] - .5), 1, 1, fill=False,
                                  edgecolor="#eb6834", lw=1.5))
        a.set_xticks(range(len(p.columns)))
        a.set_xticklabels([f"{v:g}" for v in p.columns], fontsize=5.4, rotation=90)
        a.set_yticks(range(len(p.index)))
        a.set_yticklabels([f"{v:g}" for v in p.index], fontsize=5.4)
        a.grid(False)
        a.set_title(f"{c} · {path}", fontsize=7.5, loc="left")
        if j == 0:
            a.set_ylabel(r"$Q_\mathrm{pos}$", fontsize=7)
        if i == 1:
            a.set_xlabel(r"$R_\mathrm{rate}$", fontsize=7)
fig.colorbar(im, ax=axs, label="RMSE [m]", fraction=0.025, pad=0.02)
fig.suptitle("F28  cost-weight stress (orange box = best cell of that panel)",
             fontsize=9, x=0.02, ha="left", y=0.99)
fig.savefig(f"{FIG}/F28_cost_mismatch_hand.png", bbox_inches="tight", dpi=200)
CH.to_csv(f"{FIG}/F28_cost_mismatch_hand.csv", index=False)
plt.close(fig)

print("\n  ANALYSIS -- how much of the result is the weight choice:")
for _, r in SP.iterrows():
    print(f"    {r.path:7s} {r.ctrl:9s}: {r.best:.4f} .. {r.worst:.4f} m "
          f"-> {r.ratio:5.1f}x from the weights alone")


# %% [markdown]
# ## 3 — the learned arms against the same plant mismatch
#
# The same six axes and three paths as §1, now for AC-MPC and the adaptive arm.
# These have **no hand-chosen stage cost to stress**, which is the point of §8.3,
# so the plant axis is the only one they can be swept on.
#
# The adaptive arm carries its RDP, so its observation (and, for variants B/C,
# its prediction dynamics) see the *estimate*. A parameter mismatch is exactly
# the condition the estimator is supposed to notice, which makes this the
# sharpest test of whether estimating beats averaging.

# %%
S.section(3, "model mismatch, learned", "AC-MPC and Adaptive AC-MPC on the same "
          "six axes x three paths", produces="sensitivity/model_learned.csv, F29")

acm = X.load_ckpt("acmpc", "model.pkl")
est = X.load_ckpt("acmpc_adaptive", "estimators", "all.pkl")
ckv = None
for v in ("C", "B", "A"):
    ckv = X.load_ckpt("acmpc_adaptive", "variants", f"{v}.pkl")
    if ckv is not None:
        break

LEARNED_OK = acm is not None
ADAPT_OK = ckv is not None and est is not None
if not LEARNED_OK:
    print("  SKIP AC-MPC: acmpc/model.pkl missing -- run Notebook 3")
if not ADAPT_OK:
    print("  SKIP Adaptive: acmpc_adaptive checkpoints missing -- run Notebook 5")

ML = pd.DataFrame()
if LEARNED_OK:
    if ADAPT_OK:
        p_sel, sc_sel, Hsel = A.rebuild_rdp(est[est["__selected__"]])
    rows = []
    for path in PATHS:
        for axis in AXES:
            for lv in LEVELS[axis]:
                env = mismatch_env(axis, lv, path)
                st = ev(env, X.ctrl_from_actor(acm["actor"],
                                               dict(acm["cfg"], n_diff=1)))
                rows.append(dict(path=path, axis=axis, level=lv, ctrl="AC-MPC N=1",
                                 rmse=st["rmse"], sat=st["sat"], crash=st["crash"],
                                 bound_frac=st["bound_frac"]))
                if not ADAPT_OK:
                    continue
                aenv = mismatch_env(axis, lv, path, adapt=True, H=Hsel,
                                    oracle=bool(ckv.get("to_obs", True)))
                aenv.attach(p_sel, sc_sel)
                actrl = X.ctrl_from_actor(
                    ckv["actor"], dict(ckv["cfg"], n_diff=1),
                    dmod_fn=(lambda e: e.dmod()) if ckv.get("to_model") else None)
                st = ev(aenv, actrl)
                rows.append(dict(path=path, axis=axis, level=lv,
                                 ctrl="Adaptive AC-MPC N=1", rmse=st["rmse"],
                                 sat=st["sat"], crash=st["crash"],
                                 bound_frac=st["bound_frac"]))
    ML = pd.DataFrame(rows)
    ML.to_csv(X.apath("sensitivity", "model_learned.csv"), index=False)

    cl = sorted(ML.ctrl.unique())
    S.table(ML.pivot_table(index=["axis", "ctrl"], columns="level",
                           values="rmse").reset_index(),
            "Model mismatch, learned (RMSE [m], median over paths)",
            csv=("sensitivity", "model_learned_pivot.csv"))
    bl = pd.DataFrame([dict(axis=a, **{c: _breakpoint(ML, c, a) for c in cl})
                       for a in AXES])
    S.table(bl, "Break factor, learned (first level at 3x nominal RMSE)",
            csv=("sensitivity", "model_learned_breakpoints.csv"))

    COL = {"AC-MPC N=1": "#1baf7a", "Adaptive AC-MPC N=1": "#4a3aa7"}
    MKR = {"AC-MPC N=1": "^", "Adaptive AC-MPC N=1": "D"}
    LS = {"AC-MPC N=1": "-", "Adaptive AC-MPC N=1": "--"}
    fig, axs = plt.subplots(len(PATHS), len(AXES),
                            figsize=(2.05 * len(AXES), 1.95 * len(PATHS)),
                            sharex="col")
    for i, path in enumerate(PATHS):
        for j, axis in enumerate(AXES):
            a = axs[i, j]
            for c in cl:
                d = ML[(ML.path == path) & (ML.axis == axis) &
                       (ML.ctrl == c)].sort_values("level")
                a.plot(d.level, d.rmse, color=COL[c], ls=LS[c], marker=MKR[c],
                       ms=3, lw=1.4, label=c)
            a.axvline(1.0, color="#8a8a84", lw=0.7, ls=(0, (3, 2)))
            a.set_yscale("log")
            a.grid(color="#e6e6e2", lw=0.5)
            for sp_ in ("top", "right"):
                a.spines[sp_].set_visible(False)
            if i == 0:
                a.set_title(rf"$\lambda_{{{axis}}}$", fontsize=8)
            if j == 0:
                a.set_ylabel(f"{path}\nRMSE [m]", fontsize=7)
            if i == len(PATHS) - 1:
                a.set_xlabel("factor", fontsize=7)
    axs[0, 0].legend(fontsize=5.2)
    fig.suptitle("F29  learned controllers against plant mismatch",
                 fontsize=9, x=0.02, ha="left", y=1.005)
    fig.savefig(f"{FIG}/F29_model_mismatch_learned.png", bbox_inches="tight", dpi=200)
    ML.to_csv(f"{FIG}/F29_model_mismatch_learned.csv", index=False)
    plt.close(fig)

    if ADAPT_OK:
        print("\n  ANALYSIS -- does the estimator pay on a mismatched plant?")
        print("    (negative = the adaptive arm is better on that axis)")
        for a_ in AXES:
            g = ML[ML.axis == a_].groupby("ctrl").rmse.median()
            if len(g) == 2:
                d_ = float(g["Adaptive AC-MPC N=1"] - g["AC-MPC N=1"])
                print(f"    lambda_{a_:4s}: {d_:+.4f} m")


# %% [markdown]
# ## 4 — the adaptive arm against horizon length
#
# **This one requires a retrain per horizon.** `costmap_head` emits
# `N × REP_DIM[rep]` numbers and reshapes to `(B, N, REP_DIM)`, so an actor
# trained at `N=1` has no cost map for stages 2…N and cannot simply be evaluated
# at a longer horizon. Reusing the `N=1` actor would silently reinterpret its 26
# outputs as stage 1 of a 3-stage problem, which is not the same object.
#
# Read against NB1's horizon sweep for the *tuned* NMPC, and against the p95
# latency, since a horizon is only admissible if it fits the 20 ms period.

# %%
S.section(4, "adaptive horizon sweep", "train the adaptive arm at each N, then "
          "evaluate -- the head shape makes a retrain mandatory",
          produces="sensitivity/adaptive_horizon.csv, F30")

HORIZONS = (1, 2, 3, 5)
H_ITERS = max(S.CFG["iters_sweep"] // 2, 4)
S2 = __import__("study_moderate").moderate(S.disturbed_spec(), wind=(0.0, 2.0))

rows = []
for N in HORIZONS:
    cfg = dict(N=N, rep="diag", n_iter=NIT, n_diff=2, hid=S.CFG["hid"],
               minib=S.CFG["minib"], epochs=S.CFG["epochs"], sigma=0.05,
               dist_label=f"adaptive N={N}")
    env = A.AdaptEnv(S.CFG["n_env"], 5, 200, S2, "central", 0.0, oracle=True,
                     wrench_dr=A.WRENCH_DR_START, paths=("circle", "fig8"),
                     task="stabilize")
    actor, _, log = X.train_ppo(None, cfg, seed=5, iters=H_ITERS,
                                T_rollout=S.CFG["T_rollout"], env=env, verbose=False)
    lat = None
    for path in PATHS:
        eenv = A.AdaptEnv(N_EV, 4242, 100000, S.nominal_spec(speed=SPEED),
                          "central", 0.0, paths=(path,), oracle=True)
        st = ev(eenv, X.ctrl_from_actor(actor, dict(cfg, n_diff=1)))
        if lat is None:
            e1 = A.AdaptEnv(1, 4242, 100000, S.nominal_spec(speed=SPEED),
                            "central", 0.0, paths=(path,), oracle=True)
            lat = X.solve_latency_ms(X.ctrl_from_actor(actor, dict(cfg, n_diff=1)),
                                     e1, T=25)
        rows.append(dict(N=N, path=path, rmse=st["rmse"], sat=st["sat"],
                         crash=st["crash"], bound_frac=st["bound_frac"],
                         landed=int(log.landed.sum()) if "landed" in log else -1,
                         final_reward=float(log.reward.tail(3).mean()),
                         ms_median=lat["median"], ms_p95=lat["p95"]))
AH = pd.DataFrame(rows)
AH.to_csv(X.apath("sensitivity", "adaptive_horizon.csv"), index=False)
S.table(AH.pivot_table(index="N", columns="path", values="rmse").reset_index(),
        "Adaptive AC-MPC: RMSE [m] against horizon",
        note=f"trained {H_ITERS} iterations per horizon; a retrain is mandatory "
             f"because the cost-map head emits N x REP_DIM numbers",
        csv=("sensitivity", "adaptive_horizon_pivot.csv"))

fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
for k, path in enumerate(PATHS):
    d = AH[AH.path == path].sort_values("N")
    axs[0].plot(d.N, d.rmse, color=["#2a78d6", "#eb6834", "#1baf7a"][k],
                ls=["-", "--", "-."][k], marker=["o", "s", "^"][k], ms=4, label=path)
axs[0].set_yscale("log")
axs[0].set_title("(a) accuracy vs horizon", loc="left", fontsize=8.5)
axs[0].set_xlabel("horizon N"); axs[0].set_ylabel("RMSE [m]")
axs[0].legend(fontsize=6)
lat = AH.groupby("N")[["ms_median", "ms_p95"]].first().reset_index()
axs[1].plot(lat.N, lat.ms_median, "-", color="#2a78d6", marker="o", label="median")
axs[1].plot(lat.N, lat.ms_p95, "--", color="#eb6834", marker="s", label="p95")
axs[1].axhline(20.0, color="#8a8a84", lw=0.9, ls=(0, (4, 2)))
axs[1].annotate("20 ms period", (lat.N.min(), 20.0), xytext=(1, 2),
                textcoords="offset points", fontsize=5.8, color="#52514e")
axs[1].set_yscale("log")
axs[1].set_title("(b) single-vehicle solve latency", loc="left", fontsize=8.5)
axs[1].set_xlabel("horizon N"); axs[1].set_ylabel("ms")
axs[1].legend(fontsize=6)
for a in axs:
    a.grid(color="#e6e6e2", lw=0.5)
    for sp_ in ("top", "right"):
        a.spines[sp_].set_visible(False)
fig.suptitle("F30  adaptive AC-MPC against horizon length (retrained per N)",
             fontsize=9, x=0.02, ha="left", y=1.02)
fig.savefig(f"{FIG}/F30_adaptive_horizon.png", bbox_inches="tight", dpi=200)
AH.to_csv(f"{FIG}/F30_adaptive_horizon.csv", index=False)
plt.close(fig)

best = AH.groupby("N").rmse.median().idxmin()
adm = AH[AH.ms_p95 <= 20.0].N.max() if (AH.ms_p95 <= 20.0).any() else None
print(f"\n  ANALYSIS.  Best horizon by median RMSE: N = {best}.")
print(f"  Largest horizon inside the 20 ms period: N = {adm}.")
print(f"  Trained {H_ITERS} iterations per horizon -- a longer horizon is a "
      f"bigger optimisation, so a flat or rising curve at this budget is a "
      f"statement about the budget as much as about the horizon.")


# %% [markdown]
# ## 5 — one picture: what is each controller fragile to?
#
# Degradation factor per plant axis, every controller that has data, on one
# scale. A bar is `median RMSE over the swept range / RMSE at nominal`, so 1.0
# means "did not notice" and large means "this is the axis that breaks it".

# %%
S.section(5, "fragility summary", "degradation factor per axis, all controllers",
          produces="sensitivity/fragility.csv, F31")

ALL = pd.concat([MH, ML], ignore_index=True) if len(ML) else MH.copy()
rows = []
for c in sorted(ALL.ctrl.unique()):
    for a_ in AXES:
        d = ALL[(ALL.ctrl == c) & (ALL.axis == a_)].groupby("level").rmse.median()
        if 1.0 in d.index and np.isfinite(d.loc[1.0]) and d.loc[1.0] > 0:
            rows.append(dict(ctrl=c, axis=a_, nominal=float(d.loc[1.0]),
                             worst=float(d.max()),
                             factor=float(d.max() / d.loc[1.0])))
FR = pd.DataFrame(rows)
FR.to_csv(X.apath("sensitivity", "fragility.csv"), index=False)
S.table(FR.pivot_table(index="ctrl", columns="axis", values="factor").reset_index(),
        "Fragility: worst RMSE over the swept range, divided by nominal",
        note="1.0 = the sweep never moved it; the axis with the largest entry is "
             "what that controller is actually fragile to",
        csv=("sensitivity", "fragility_pivot.csv"))

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100", "#e87ba4"]
piv = FR.pivot_table(index="axis", columns="ctrl", values="factor").reindex(AXES)
fig, ax = plt.subplots(figsize=(6.4, 2.9))
xs = np.arange(len(piv))
w = 0.8 / max(len(piv.columns), 1)
for j, c in enumerate(piv.columns):
    v = piv[c].values
    ax.bar(xs + (j - (len(piv.columns) - 1) / 2) * w, v, w * 0.9, label=c,
           color=PAL[j % len(PAL)], edgecolor="white", linewidth=0.5)
    for x, y in zip(xs + (j - (len(piv.columns) - 1) / 2) * w, v):
        if np.isfinite(y):
            ax.annotate(f"{y:.1f}", (x, y), ha="center", va="bottom", fontsize=5.0,
                        color="#52514e", xytext=(0, 1.6), textcoords="offset points",
                        rotation=90)
ax.axhline(1.0, color="#8a8a84", lw=0.8, ls=(0, (3, 2)))
ax.set_yscale("log")
ax.margins(y=0.25)
ax.set_xticks(xs)
ax.set_xticklabels([rf"$\lambda_{{{a}}}$" for a in piv.index])
ax.set_ylabel("worst / nominal RMSE")
ax.set_title("F31  what each controller is fragile to (1.0 = unmoved by the sweep)",
             loc="left", fontsize=8.5)
ax.grid(color="#e6e6e2", lw=0.5)
for sp_ in ("top", "right"):
    ax.spines[sp_].set_visible(False)
ax.legend(fontsize=5.8, ncol=3)
fig.savefig(f"{FIG}/F31_fragility.png", bbox_inches="tight", dpi=200)
FR.to_csv(f"{FIG}/F31_fragility.csv", index=False)
plt.close(fig)

print("\n  ANALYSIS -- each controller's worst axis:")
for c in sorted(FR.ctrl.unique()):
    d = FR[FR.ctrl == c].sort_values("factor", ascending=False)
    r = d.iloc[0]
    print(f"    {c:22s} worst on lambda_{r.axis:4s} ({r.factor:6.2f}x), "
          f"best on lambda_{d.iloc[-1].axis:4s} ({d.iloc[-1].factor:.2f}x)")

S.checkpoint(model_hand=("sensitivity", "model_hand.csv"),
             cost_hand=("sensitivity", "cost_hand.csv"),
             model_learned=("sensitivity", "model_learned.csv"),
             adaptive_horizon=("sensitivity", "adaptive_horizon.csv"),
             fragility=("sensitivity", "fragility.csv"))
print("\nNotebook 8 complete.")
