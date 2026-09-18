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
    "m": (0.55, 0.70, 0.85, 1.00, 1.20, 1.45, 1.75, 2.10),
    "J": (0.40, 0.60, 0.80, 1.00, 1.30, 1.70, 2.20, 2.80),
}
AXES = tuple(LEVELS)
#: The mismatch sweeps run on **fig8 only**.  It is the path that is partly at
#: the feasibility cap (NB1: capped on 6.2 % of radii, against square's 100 % and
#: circle's none), so it stresses a controller without the collective already
#: being pinned -- on square the sweep would largely measure the cap.  The cost
#: sweep of §2 still runs on all three, because there the path is the variable
#: of interest rather than a confound.
MIS_PATHS = ("fig8",)


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
S.section(1, "model mismatch, hand-built", "LQR and NMPC N=1 off-nominal in mass "
          "and inertia, on fig8", produces="sensitivity/model_hand.csv, F27")

rows = []
for path in MIS_PATHS:
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

# one table PER AXIS: the axes have different level grids, so a shared `level`
# column would be half NaN and unreadable
for _a in AXES:
    _p = MH[MH.axis == _a].pivot_table(index="ctrl", columns="level",
                                       values="rmse").reset_index()
    S.table(_p, f"Model mismatch, hand-built -- lambda_{_a} (RMSE [m])",
            note="columns are the multiplicative factor on that parameter; "
                 "1.0 is nominal",
            csv=("sensitivity", f"model_hand_{_a}.csv"))


def _breakpoint(df, ctrl, axis, thresh=3.0):
    """Factor at which RMSE first exceeds `thresh` x **its own** nominal.

    A RELATIVE measure, and it must be read as one: a controller that starts
    four times more accurate trips a 3x threshold four times sooner while still
    being ahead in absolute terms.  Measured here, NMPC N=1 sits at 0.031 m at
    nominal against LQR's 0.127 m, so "NMPC breaks at 0.55x mass" means it
    reached 0.093 m there -- where LQR was at 0.168 m and therefore still worse.
    Read this column beside :func:`_crossover`, never alone.
    """
    d = df[(df.ctrl == ctrl) & (df.axis == axis)].groupby("level").rmse.median()
    if 1.0 not in d.index:
        return np.nan
    base = d.loc[1.0]
    bad = d[d > thresh * base]
    return float(bad.index.min()) if len(bad) else np.nan


def _crossover(df, a, b, axis, side):
    """Factor at which `a` stops being better than `b`, in ABSOLUTE RMSE.

    The question the relative break factor cannot answer: a steeper slope from a
    lower starting point may never actually lose.  `side` is 'up' or 'down' --
    mismatch has two directions and a controller can be robust to one and not
    the other.
    """
    da = df[(df.ctrl == a) & (df.axis == axis)].groupby("level").rmse.median()
    db = df[(df.ctrl == b) & (df.axis == axis)].groupby("level").rmse.median()
    lv = sorted(set(da.index) & set(db.index))
    lv = [x for x in lv if (x >= 1.0 if side == "up" else x <= 1.0)]
    if side == "down":
        lv = lv[::-1]
    for x in lv:
        if da.loc[x] > db.loc[x]:
            return float(x)
    return np.nan


bp = pd.DataFrame([
    dict(axis=a,
         nominal_LQR=float(MH[(MH.ctrl == "LQR") & (MH.axis == a) &
                              (MH.level == 1.0)].rmse.median()),
         nominal_NMPC=float(MH[(MH.ctrl == "NMPC N=1") & (MH.axis == a) &
                               (MH.level == 1.0)].rmse.median()),
         break_LQR=_breakpoint(MH, "LQR", a),
         break_NMPC=_breakpoint(MH, "NMPC N=1", a),
         NMPC_loses_above=_crossover(MH, "NMPC N=1", "LQR", a, "up"),
         NMPC_loses_below=_crossover(MH, "NMPC N=1", "LQR", a, "down"))
    for a in AXES])
S.table(bp, "Where each hand-built arm gives way",
        note="`break_*` is RELATIVE (3x that controller's OWN nominal) and a more "
             "accurate controller trips it sooner while still leading; "
             "`NMPC_loses_*` is the ABSOLUTE crossover, the factor at which NMPC "
             "N=1 stops beating LQR.  NaN in the crossover columns means NMPC "
             "never lost on that side of nominal.",
        csv=("sensitivity", "model_hand_breakpoints.csv"))

fig, axs = plt.subplots(len(MIS_PATHS), len(AXES),
                        figsize=(3.1 * len(AXES), 2.6 * len(MIS_PATHS)),
                        squeeze=False)
for i, path in enumerate(MIS_PATHS):
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
            a.set_title(rf"$\lambda_{{{axis}}}$", fontsize=8.5, loc="left")
        if j == 0:
            a.set_ylabel("RMSE [m]", fontsize=7)
        a.set_xlabel("factor", fontsize=7)
axs[0, 0].legend(fontsize=5.6)
fig.suptitle(f"F27  hand-built controllers against plant mismatch, {MIS_PATHS[0]} "
             "(vertical line = nominal)", fontsize=9, x=0.02, ha="left", y=1.06)
fig.savefig(f"{FIG}/F27_model_mismatch_hand.png", bbox_inches="tight", dpi=200)
MH.to_csv(f"{FIG}/F27_model_mismatch_hand.csv", index=False)
plt.close(fig)

f_ = lambda v: "never in range" if not np.isfinite(v) else f"{v:g}x"
print("\n  ANALYSIS -- two different questions, two different answers:")
print("    (a) RELATIVE: 3x its own nominal.  The more accurate controller")
print("        trips this sooner even while still leading in absolute terms.")
for a_ in AXES:
    print(f"        lambda_{a_:4s}: LQR {f_(_breakpoint(MH, 'LQR', a_)):>14s}"
          f"   NMPC N=1 {f_(_breakpoint(MH, 'NMPC N=1', a_)):>14s}")
print("    (b) ABSOLUTE: the factor at which NMPC N=1 stops beating LQR.")
for a_ in AXES:
    up = _crossover(MH, "NMPC N=1", "LQR", a_, "up")
    dn = _crossover(MH, "NMPC N=1", "LQR", a_, "down")
    print(f"        lambda_{a_:4s}: above nominal {f_(up):>14s}"
          f"   below nominal {f_(dn):>14s}")
print("    The online solve degrades FASTER from a lower base.  Whether that")
print("    means it is less robust depends on which question is being asked,")
print("    and the two columns disagree over much of the range.")


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
          "two axes, on fig8", produces="sensitivity/model_learned.csv, F29")

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
    for path in MIS_PATHS:
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
    for _a in AXES:
        _p = ML[ML.axis == _a].pivot_table(index="ctrl", columns="level",
                                           values="rmse").reset_index()
        S.table(_p, f"Model mismatch, learned -- lambda_{_a} (RMSE [m])",
                csv=("sensitivity", f"model_learned_{_a}.csv"))
    bl = pd.DataFrame([dict(axis=a, **{c: _breakpoint(ML, c, a) for c in cl})
                       for a in AXES])
    S.table(bl, "Break factor, learned (first level at 3x nominal RMSE)",
            csv=("sensitivity", "model_learned_breakpoints.csv"))

    COL = {"AC-MPC N=1": "#1baf7a", "Adaptive AC-MPC N=1": "#4a3aa7"}
    MKR = {"AC-MPC N=1": "^", "Adaptive AC-MPC N=1": "D"}
    LS = {"AC-MPC N=1": "-", "Adaptive AC-MPC N=1": "--"}
    fig, axs = plt.subplots(len(MIS_PATHS), len(AXES),
                            figsize=(3.1 * len(AXES), 2.6 * len(MIS_PATHS)),
                            squeeze=False)
    for i, path in enumerate(MIS_PATHS):
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
                a.set_title(rf"$\lambda_{{{axis}}}$", fontsize=8.5, loc="left")
            if j == 0:
                a.set_ylabel("RMSE [m]", fontsize=7)
            a.set_xlabel("factor", fontsize=7)
    axs[0, 0].legend(fontsize=5.2)
    fig.suptitle(f"F29  learned controllers against plant mismatch, "
                 f"{MIS_PATHS[0]}", fontsize=9, x=0.02, ha="left", y=1.06)
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
# ## 4 — how much history does the RDP need?
#
# The predictor's horizon is its **causal window length** $H$: how many past
# frames of (6.2) it sees before predicting the wrench. `H_BENCH = (16, 32, 64,
# 128)` has been in `adaptive_core_jax` since the start and was never swept — NB5
# compares the four encoders at the single value $H=64$.
#
# $H$ is not free. It sets the deployment ring-buffer length, the warm-up before
# the estimate is usable, and the inference latency that has to fit inside the
# 20 ms period. It also sizes two of the architectures: `tcn_blocks_for(H)` and
# `cnn_layers_for(H)` pick the smallest depth whose receptive field covers the
# whole window, so a TCN at $H=128$ is a deeper network than at $H=16$, not the
# same one fed more input.
#
# The question this answers: **is a long window buying accuracy, or is the
# residual essentially memoryless?**

# %%
S.section(4, "RDP window length", "sweep H over H_BENCH for every encoder -- "
          "accuracy, cost and warm-up against how much history is kept",
          produces="sensitivity/rdp_window.csv, F30")

import study_moderate as M                                          # noqa: E402

H_LIST = A.H_BENCH
S2 = M.moderate(S.disturbed_spec(), wind=(0.0, 2.0))
N_EP_D = 8 if S.SCALE == "smoke" else 24
T_D = 200 if S.SCALE == "smoke" else 500
pilot = X.make_nmpc_ctrl(N=1, n_iter=NIT)


def collect(scen, level, seed, n, T):
    """Fly under a known disturbance, log the causal frame and the truth.

    Generated ONCE at the largest H and re-windowed per H, so every row of the
    sweep sees the same flights -- otherwise a window-length comparison would
    also be a different-data comparison.
    """
    env = A.AdaptEnv(n, seed, T + 10, S2, scen, level, paths=("circle", "fig8"),
                     wrench_dr=A.WRENCH_DR_START, task="stabilize", H=max(H_LIST))
    pilot.bind_env(env)
    F, Y, EP = [], [], []
    o, e, xr = env.obs()
    for _ in range(T):
        _, uref, _ = env.ref_now()
        u, _ = pilot(o, e, xr, uref=uref)
        F.append(np.asarray(env.frame26()))
        Y.append(np.asarray(env.d_truth()))
        EP.append(np.asarray(env.n_step))
        env.step(u)
        o, e, xr = env.obs()
    return np.array(F), np.array(Y), np.array(EP)


print(f"  generating data once: {N_EP_D} vehicles x {T_D} steps x 3 scenarios")
FR_, YR_, EID_ = [], [], []
for scen, lvl in (("central", 0.15), ("asym", 0.07), ("slung", 10.0)):
    f, y, ep = collect(scen, lvl, 31, N_EP_D, T_D)
    FR_.append(f); YR_.append(y)
    EID_.append(np.cumsum(np.diff(ep, axis=0, prepend=ep[:1]) < 0, axis=0))

rows = []
for H in H_LIST:
    Ws, Ys = [], []
    for f, y, eid in zip(FR_, YR_, EID_):
        w, yy = A.make_windows(f, y, H, ep_id=eid)
        Ws.append(w); Ys.append(yy)
    W, Y = np.concatenate(Ws), np.concatenate(Ys)
    blocks = np.arange(W.shape[0]) // max(T_D - H, 1)
    ub = np.unique(blocks)
    tr_b, va_b, te_b = A.split_by_episode(len(ub), seed=0)
    m_tr = np.isin(blocks, ub[tr_b]); m_va = np.isin(blocks, ub[va_b])
    m_te = np.isin(blocks, ub[te_b])
    for kind in A.ENCODERS:
        p, sc, _ = A.train_rdp(__import__("jax").random.PRNGKey(7), kind,
                               W[m_tr], Y[m_tr], W[m_va], Y[m_va], H=H,
                               epochs=S.CFG["est_epochs"], batch=S.CFG["est_batch"],
                               verbose=False)
        pred = A.rdp_predict(p, sc, W[m_te])
        r2 = A.r2_score(Y[m_te], pred)
        lat = A.rdp_latency_ms(p, sc, H)
        rows.append(dict(H=H, encoder=kind, n_windows=int(W.shape[0]),
                         params=A.rdp_param_count(p),
                         r2_overall=float(np.mean(r2)),
                         r2_force=float(np.mean(r2[:3])),
                         r2_moment=float(np.mean(r2[3:])),
                         lat_median_ms=lat["median"], lat_p95_ms=lat["p95"],
                         admissible=bool(lat["p95"] <= 20.0),
                         warmup_ms=float(H * X.P.dt_c * 1e3)))
        print(f"    H={H:4d}  {kind:5s}  R2 {np.mean(r2):+.4f}  "
              f"p95 {lat['p95']:.2f} ms  warm-up {H * X.P.dt_c * 1e3:.0f} ms")
RW = pd.DataFrame(rows)
RW.to_csv(X.apath("sensitivity", "rdp_window.csv"), index=False)
S.table(RW.pivot_table(index="H", columns="encoder", values="r2_overall").reset_index(),
        "RDP accuracy against window length (R^2 overall)",
        note="H is also the deployment warm-up: the estimate is unusable for the "
             "first H frames (H*dt_c ms) after every respawn",
        csv=("sensitivity", "rdp_window_pivot.csv"))

fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.6))
PAL = {"GRU": "#2a78d6", "LSTM": "#eb6834", "TCN": "#1baf7a", "CNN": "#eda100"}
MKS = {"GRU": "o", "LSTM": "s", "TCN": "^", "CNN": "D"}
LSS = {"GRU": "-", "LSTM": "--", "TCN": "-.", "CNN": ":"}
for enc in A.ENCODERS:
    d = RW[RW.encoder == enc].sort_values("H")
    if not len(d):
        continue
    axs[0].plot(d.H, d.r2_overall, color=PAL[enc], ls=LSS[enc], marker=MKS[enc],
                ms=4, label=enc)
    axs[1].plot(d.H, d.r2_force, color=PAL[enc], ls=LSS[enc], marker=MKS[enc], ms=4)
    axs[1].plot(d.H, d.r2_moment, color=PAL[enc], ls=LSS[enc], marker=MKS[enc],
                ms=4, mfc="white", alpha=0.75)
    axs[2].plot(d.H, d.lat_p95_ms, color=PAL[enc], ls=LSS[enc], marker=MKS[enc],
                ms=4, label=enc)
axs[0].set_title("(a) overall accuracy", loc="left", fontsize=8.5)
axs[0].set_ylabel(r"$R^2$")
axs[0].legend(fontsize=6)
axs[1].set_title("(b) force (filled) vs moment (hollow)", loc="left", fontsize=8.5)
axs[1].set_ylabel(r"$R^2$")
axs[2].axhline(20.0, color="#8a8a84", lw=0.9, ls=(0, (4, 2)))
axs[2].annotate("20 ms period", (min(H_LIST), 20.0), xytext=(1, 2),
                textcoords="offset points", fontsize=5.8, color="#52514e")
axs[2].set_yscale("log")
axs[2].set_title("(c) inference latency, one window", loc="left", fontsize=8.5)
axs[2].set_ylabel("p95 [ms]")
for a in axs:
    a.set_xscale("log", base=2)
    a.set_xticks(list(H_LIST))
    a.set_xticklabels([str(h) for h in H_LIST])
    a.set_xlabel("window length H [frames]")
    a.grid(color="#e6e6e2", lw=0.5)
    for sp_ in ("top", "right"):
        a.spines[sp_].set_visible(False)
fig.suptitle("F30  how much history the RDP needs (H is also the warm-up: "
             f"{min(H_LIST)*X.P.dt_c*1e3:.0f}-{max(H_LIST)*X.P.dt_c*1e3:.0f} ms)",
             fontsize=9, x=0.02, ha="left", y=1.03)
fig.savefig(f"{FIG}/F30_rdp_window.png", bbox_inches="tight", dpi=200)
RW.to_csv(f"{FIG}/F30_rdp_window.csv", index=False)
plt.close(fig)

print("\n  ANALYSIS -- is a longer window buying accuracy?")
for enc in A.ENCODERS:
    d = RW[RW.encoder == enc].sort_values("H")
    if len(d) < 2:
        continue
    lo, hi = d.iloc[0], d.iloc[-1]
    print(f"    {enc:5s}: R2 {lo.r2_overall:+.4f} at H={int(lo.H)} -> "
          f"{hi.r2_overall:+.4f} at H={int(hi.H)} "
          f"({hi.r2_overall - lo.r2_overall:+.4f}), p95 "
          f"{lo.lat_p95_ms:.2f} -> {hi.lat_p95_ms:.2f} ms")
_best = RW.loc[RW.r2_overall.idxmax()]
_cheap = RW[RW.r2_overall >= 0.95 * RW.r2_overall.max()].sort_values("H").iloc[0]
print(f"  Best overall: {_best.encoder} at H={int(_best.H)} "
      f"(R2 {_best.r2_overall:+.4f}).")
print(f"  Within 5 % of it at the SHORTEST window: {_cheap.encoder} at "
      f"H={int(_cheap.H)} (R2 {_cheap.r2_overall:+.4f}, warm-up "
      f"{_cheap.warmup_ms:.0f} ms) -- H is a warm-up cost after every respawn, "
      f"so the shortest adequate window is the one to deploy.")


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
             rdp_window=("sensitivity", "rdp_window.csv"),
             fragility=("sensitivity", "fragility.csv"))
print("\nNotebook 8 complete.")
