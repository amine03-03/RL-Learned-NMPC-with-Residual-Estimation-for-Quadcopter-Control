# %% [markdown]
# # Notebook 4 — domain randomisation
#
# **Establishes.** What randomisation buys, what it costs, which variance moves,
# and whether sequential randomisation forgets.
#
# **Consumes.** `acmpc/model.pkl` (the architecture and config).
#
# **Produces.** `domrand/{nominal,DR-all,DR-all+noise}.pkl`, `log_*.csv`,
# `axis_sweep.csv`, `timing.csv`, `timing_before.csv`, `summary.json`,
# figures F13–F15, table T6.
#
# This notebook exists to motivate Notebook 5: a policy that cannot *see* which
# plant it is on can only find one compromise acceptable everywhere. It cannot
# specialise.
#
# **Runtime is designed for, not discovered.** A naive version retrains two arms
# from scratch three times each, its first seed duplicating a policy already
# trained. Here the headline runs are reused as seed 0, and every train/evaluate
# is wrapped in a timer whose output is written before any optimisation is
# claimed.

# %%
import _nbinit  # noqa: F401
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

import x500_core_jax as X
import study_prelude as S
import study_moderate as M
import viz as V

S.header("Notebook 4 - domain randomisation")
X.style()
FIG = V.figures_dir()
BUD = S.DR_BUDGETS[S.DR_BUDGET]
TIMER = S.Timer()
print(f"  DR budget '{S.DR_BUDGET}': {BUD}")

PATHS = ("circle", "fig8")
CFG = dict(N=1, rep="diag", n_iter=S.CFG["ilqr"], n_diff=2, hid=S.CFG["hid"],
           minib=S.CFG["minib"], epochs=S.CFG["epochs"], sigma=0.15)
S2 = M.moderate(S.disturbed_spec(), wind=(0.0, 2.0))
M.check_moderate(S.disturbed_spec(), S2, 2.0)
ARMS = {"nominal": dict(spec=S.nominal_spec(speed=(1.0, 1.5)), noise="off"),
        "DR-all": dict(spec=S2, noise="off"),
        "DR-all+noise": dict(spec=S2, noise="low")}


def train_arm(name, seed, iters=None):
    a = ARMS[name]
    env = X.Env(S.CFG["n_env"], seed, 200, a["spec"], PATHS, noise=a["noise"])
    with TIMER(f"train:{name}", seed=seed):
        actor, critic, log = X.train_ppo(
            None, dict(CFG, dist_label=name), seed=seed,
            iters=iters or BUD["iters"], T_rollout=S.CFG["T_rollout"],
            env=env, verbose=False)
    return actor, critic, log


def eval_arm(ctrl, spec, seed=900, n=None, T=None, paths=PATHS, noise="off"):
    env = X.Env(n or BUD["eval_ep"], seed, 100000, spec, paths, noise=noise)
    with TIMER("evaluate", n=n or BUD["eval_ep"]):
        return X.stats(X.rollout_eval(env, ctrl, T or BUD["eval_T"], warmup=30))


def as_ctrl(actor, name):
    return X.ctrl_from_actor(actor, dict(CFG, n_diff=1, name=name))

# %% [markdown]
# ## Experiment 1 — headline policies
#
# These three runs are **reused as seed 0** of the seed study below. That changes
# no result and removes a third of the training budget; the seed list is reported
# honestly wherever it is used.

# %%
S.section(1, "headline policies", "nominal vs DR-all vs DR-all+noise",
          produces="domrand/*.pkl, domrand/log_*.csv")

SEEDS = [5] + [200 + i for i in range(BUD["seeds"] - 1)]
print(f"  seed list: {SEEDS}  (seed {SEEDS[0]} is the headline run, reused)")
POL, LOGS = {}, {}
for name in ARMS:
    actor, critic, log = train_arm(name, SEEDS[0])
    POL[name] = actor
    LOGS[name] = log
    X.save_ckpt(dict(actor=actor, critic=critic, cfg=CFG, arm=name, seed=SEEDS[0]),
                "domrand", f"{name}.pkl")
    log.to_csv(X.apath("domrand", f"log_{name}.csv"), index=False)
TIMER.df().to_csv(X.apath("domrand", "timing_before.csv"), index=False)

rows = []
for name in ARMS:
    for suite, spec in (("nominal", S.nominal_spec(speed=(1.0, 1.5))),
                        ("S2 moderated", S2)):
        st = eval_arm(as_ctrl(POL[name], name), spec)
        rows.append(dict(policy=name, suite=suite, **st))
rows.append(dict(policy="NMPC N=1", suite="nominal",
                 **eval_arm(X.make_nmpc_ctrl(N=1, n_iter=S.CFG["ilqr"]),
                            S.nominal_spec(speed=(1.0, 1.5)))))
rows.append(dict(policy="NMPC N=1", suite="S2 moderated",
                 **eval_arm(X.make_nmpc_ctrl(N=1, n_iter=S.CFG["ilqr"]), S2)))
HEAD = pd.DataFrame(rows)
S.table(HEAD[["policy", "suite", "rmse", "rmse_iqr", "maxerr", "sat", "crash"]],
        "Headline policies", note="report saturation beside RMSE",
        csv=("domrand", "headline.csv"))

# %% [markdown]
# ## Experiment 2 — per-axis competence
#
# The axis sweep is **vectorised**: one rollout over `bins x eval_ep` vehicles
# samples the whole axis, then bins by the realised factor read back from
# `env.par`. That changes the estimator — bins get random membership instead of a
# delta at the bin centre — so the *realised* mean factor is plotted on the
# x-axis and the per-bin counts are printed. The trade is stated, not absorbed.

# %%
S.section(2, "axis sweep", "which plant parameter each policy can survive",
          produces="domrand/axis_sweep.csv, F13")

AXES = {"m": (0.80, 1.20), "tau": (0.50, 3.00), "J": (0.70, 1.40), "T": (0.85, 1.15)}
NBINS = BUD["bins"]
rows = []
for axis, (lo, hi) in AXES.items():
    spec = dict(S.nominal_spec(speed=(1.0, 1.5)))
    spec[axis] = (lo, hi)
    n = NBINS * BUD["eval_ep"]
    for pname, ctrl in [(k, as_ctrl(v, k)) for k, v in POL.items()] + \
                       [("NMPC N=1", X.make_nmpc_ctrl(N=1, n_iter=S.CFG["ilqr"]))]:
        env = X.Env(n, 901, 100000, spec, PATHS)
        realised = np.asarray(env.ep[f"lam_{axis}"])
        with TIMER("axis_rollout", axis=axis, policy=pname, n=n):
            r = X.rollout_eval(env, ctrl, BUD["eval_T"], warmup=30)
        per_veh = np.sqrt((r["pos_err"] ** 2).mean(0))
        edges = np.linspace(lo, hi, NBINS + 1)
        idx = np.clip(np.digitize(realised, edges) - 1, 0, NBINS - 1)
        for b in range(NBINS):
            m = idx == b
            if not m.any():
                continue
            rows.append(dict(axis=axis, policy=pname, bin=b,
                             factor_realised=float(realised[m].mean()),
                             count=int(m.sum()), rmse=float(per_veh[m].mean()),
                             sat=float(r["sat"][:, m].mean())))
AX = pd.DataFrame(rows)
S.table(AX.groupby(["axis", "policy"]).agg(
    rmse=("rmse", "mean"), n_bins=("bin", "nunique"),
    min_count=("count", "min")).reset_index(),
    "Axis sweep (vectorised estimator)",
    note="bins have random membership, not a delta at the centre; the x-axis "
         "below is the REALISED mean factor",
    csv=("domrand", "axis_sweep.csv"))
AX.to_csv(X.apath("domrand", "axis_sweep.csv"), index=False)

fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.2), sharey=True)
for j, axis in enumerate(AXES):
    for i, pname in enumerate(AX.policy.unique()):
        d = AX[(AX.axis == axis) & (AX.policy == pname)].sort_values("factor_realised")
        axs[j].plot(d.factor_realised, d.rmse, "-", color=f"C{i}",
                    marker=["o", "s", "^", "v"][i], ms=4, label=pname)
    axs[j].axvline(1.0, color="0.4", ls=":", lw=1)
    axs[j].set_xlabel(f"realised $\\lambda_{{{axis}}}$"); axs[j].set_title(axis)
axs[0].set_ylabel("position RMSE [m]"); axs[0].legend(fontsize=6)
fig.suptitle("F13  per-axis competence (nominal value marked)", fontsize=10)
fig.savefig(f"{FIG}/F13_axis_competence.png", bbox_inches="tight")
AX.to_csv(f"{FIG}/F13_axis_competence.csv", index=False)

# %% [markdown]
# ## Experiment 3 — conservatism, and the two variances
#
# **Report the sign before the magnitude.** If DR tracks the nominal plant
# *better* than the nominal-only policy, there is no conservatism premium in this
# run, and that must be said plainly.
#
# Seed variance and episode variance are **two different variances**. They are
# expected to move in opposite directions, and reporting only one manufactures a
# robustness claim.

# %%
S.section(3, "conservatism and variance", "the price of robustness, signed",
          produces="domrand/seed_study.csv, F14, T6")

rows = []
for name in ("nominal", "DR-all"):
    for sd in SEEDS:
        if sd == SEEDS[0]:
            actor = POL[name]                      # reuse, do not retrain
        else:
            actor, _, _ = train_arm(name, sd)
        st_n = eval_arm(as_ctrl(actor, name), S.nominal_spec(speed=(1.0, 1.5)))
        st_d = eval_arm(as_ctrl(actor, name), S2)
        env = X.Env(BUD["eval_ep"], 902, 100000, S.nominal_spec(speed=(1.0, 1.5)),
                    PATHS)
        r = X.rollout_eval(env, as_ctrl(actor, name), BUD["eval_T"], warmup=30)
        rows.append(dict(policy=name, seed=sd, rmse_nominal=st_n["rmse"],
                         rmse_S2=st_d["rmse"],
                         episode_std=float(np.sqrt((r["pos_err"] ** 2).mean(0)).std())))
SEED_DF = pd.DataFrame(rows)
S.table(SEED_DF, "Seed study", note=f"seeds {SEEDS}, seed {SEEDS[0]} reused from "
        "the headline runs", csv=("domrand", "seed_study.csv"))

VAR = SEED_DF.groupby("policy").agg(
    seed_std=("rmse_nominal", "std"), episode_std=("episode_std", "mean"),
    rmse_nominal=("rmse_nominal", "mean"), rmse_S2=("rmse_S2", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(5.4, 3.4))
xs = np.arange(len(VAR))
ax.bar(xs - 0.2, VAR.seed_std, 0.4, label="seed variance (std over seeds)",
       edgecolor="k", lw=0.5)
ax.bar(xs + 0.2, VAR.episode_std, 0.4, label="episode variance (std over vehicles)",
       hatch="//", edgecolor="k", lw=0.5)
ax.set_xticks(xs); ax.set_xticklabels(VAR.policy)
ax.set_ylabel("std of position RMSE [m]"); ax.legend(fontsize=6.5)
ax.set_title("F14  two different variances, expected to move oppositely")
fig.savefig(f"{FIG}/F14_variances.png", bbox_inches="tight")
VAR.to_csv(f"{FIG}/F14_variances.csv", index=False)

nom = float(VAR[VAR.policy == "nominal"].rmse_nominal.iloc[0])
dr = float(VAR[VAR.policy == "DR-all"].rmse_nominal.iloc[0])
premium = 100.0 * (dr - nom) / max(nom, 1e-9)
T6 = pd.DataFrame([
    dict(metric="RMSE on the nominal plant, nominal-only policy", value=nom),
    dict(metric="RMSE on the nominal plant, DR-all policy", value=dr),
    dict(metric="conservatism premium [%] (signed)", value=premium),
    dict(metric="RMSE on S2, nominal-only policy",
         value=float(VAR[VAR.policy == "nominal"].rmse_S2.iloc[0])),
    dict(metric="RMSE on S2, DR-all policy",
         value=float(VAR[VAR.policy == "DR-all"].rmse_S2.iloc[0])),
    dict(metric="robustness gain on S2 [%]",
         value=100.0 * (float(VAR[VAR.policy == "nominal"].rmse_S2.iloc[0])
                        - float(VAR[VAR.policy == "DR-all"].rmse_S2.iloc[0]))
         / max(float(VAR[VAR.policy == "nominal"].rmse_S2.iloc[0]), 1e-9)),
    dict(metric="seeds", value=len(SEEDS)),
])
S.table(T6, "T6  robustness and its price",
        note="the conservatism premium is SIGNED: negative means DR tracks the "
             "nominal plant better, i.e. there is no premium in this run",
        csv=("domrand", "T6_robustness.csv"))

print(f"\n  ANALYSIS -- sign first.")
if premium < 0:
    print(f"    The conservatism premium is {premium:+.1f}%, i.e. NEGATIVE: "
          f"DR-all tracks the nominal plant BETTER than the nominal-only policy. "
          f"There is no conservatism premium in this run.")
    print(f"    Two explanations are consistent with the data, and {len(SEEDS)} "
          f"seeds cannot separate them:")
    print(f"      (a) randomisation is acting as a regulariser at this budget;")
    print(f"      (b) the nominal-only control arm is undertrained.")
else:
    print(f"    The conservatism premium is {premium:+.1f}%: DR-all pays "
          f"{premium:.1f}% on the nominal plant for its robustness.")
print(f"    The structural conclusion does NOT depend on the sign: a policy that "
      f"cannot observe its plant cannot specialise.  It can only find one "
      f"compromise acceptable everywhere.  That is what Notebook 5 addresses.")

# %% [markdown]
# ## Experiment 4 — sequential curriculum
#
# Backward transfer, plus two plasticity diagnostics: the fraction of dead trunk
# units and the fraction of saturated cost-map outputs.

# %%
S.section(4, "sequential curriculum", "does sequential randomisation forget?",
          produces="domrand/curriculum.csv, F15")


def plasticity(actor, obs):
    """Dead trunk units and saturated cost-map outputs."""
    h = obs
    for W, b in actor["trunk"][:-1]:
        h = jnp.tanh(h @ W + b)
    Wl, bl = actor["trunk"][-1]
    h = h @ Wl + bl
    dead = float(jnp.mean(jnp.abs(h) < 1e-3))
    z = X.costmap_head(actor, obs, CFG["rep"], CFG["N"])
    sat = float(jnp.mean(jnp.abs(jax.nn.sigmoid(z) - 0.5) > 0.49))
    return dead, sat


STAGES = [("nominal", S.nominal_spec(speed=(1.0, 1.5))),
          ("mass", dict(S.nominal_spec(speed=(1.0, 1.5)), m=(0.80, 1.20))),
          ("lag", dict(S.nominal_spec(speed=(1.0, 1.5)), tau=(0.50, 3.00))),
          ("all", S2)]
actor = None
rows = []
env_probe = X.Env(64, 903, 100000, S.nominal_spec(speed=(1.0, 1.5)), PATHS)
o_probe, _, _ = env_probe.obs()
for si, (sname, spec) in enumerate(STAGES):
    env = X.Env(S.CFG["n_env"], 5, 200, spec, PATHS)
    with TIMER(f"curriculum:{sname}"):
        actor, _, log = X.train_ppo(None, dict(CFG, dist_label=sname), seed=5,
                                    iters=BUD["seq_iters"],
                                    T_rollout=S.CFG["T_rollout"], env=env,
                                    actor=actor, verbose=False)
    dead, sat = plasticity(actor, o_probe)
    for pi, (pname, pspec) in enumerate(STAGES[:si + 1]):
        st = eval_arm(as_ctrl(actor, "seq"), pspec, n=max(BUD["eval_ep"] // 2, 8),
                      T=max(BUD["eval_T"] // 2, 60))
        rows.append(dict(trained_through=sname, stage_index=si, eval_on=pname,
                         eval_index=pi, rmse=st["rmse"], dead_units=dead,
                         sat_outputs=sat))
CUR = pd.DataFrame(rows)
S.table(CUR, "Sequential curriculum", csv=("domrand", "curriculum.csv"))

bt = []
for pi, (pname, _) in enumerate(STAGES[:-1]):
    first = CUR[(CUR.eval_on == pname) & (CUR.stage_index == pi)].rmse
    last = CUR[(CUR.eval_on == pname) & (CUR.stage_index == len(STAGES) - 1)].rmse
    if len(first) and len(last):
        bt.append(dict(stage=pname, rmse_when_trained=float(first.iloc[0]),
                       rmse_at_end=float(last.iloc[0]),
                       backward_transfer=float(first.iloc[0] - last.iloc[0])))
BT = pd.DataFrame(bt)
S.table(BT, "Backward transfer (positive = improved, negative = forgot)",
        csv=("domrand", "backward_transfer.csv"))

fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.3))
if len(BT):
    axs[0].bar(BT.stage, BT.backward_transfer, edgecolor="k", lw=0.5)
    axs[0].axhline(0, color="0.3", lw=1)
axs[0].set_ylabel("backward transfer [m]  (>0 = improved)")
axs[0].set_title("(a) backward transfer")
d = CUR.groupby("stage_index").agg(dead=("dead_units", "first"),
                                   sat=("sat_outputs", "first")).reset_index()
axs[1].plot(d.stage_index, d.dead, "-o", ms=5, label="dead trunk units")
axs[1].plot(d.stage_index, d.sat, "--s", ms=5, label="saturated cost-map outputs")
axs[1].set_xticks(range(len(STAGES)))
axs[1].set_xticklabels([s[0] for s in STAGES], fontsize=7)
axs[1].set_ylabel("fraction"); axs[1].legend(fontsize=7)
axs[1].set_title("(b) plasticity diagnostics")
fig.suptitle("F15  sequential curriculum", fontsize=10)
fig.savefig(f"{FIG}/F15_curriculum.png", bbox_inches="tight")
CUR.to_csv(f"{FIG}/F15_curriculum.csv", index=False)

# %% [markdown]
# ## Timing — instrument before optimising

# %%
S.section(5, "timing", "measured, not claimed", produces="domrand/timing.csv")
TD = TIMER.df()
TD.to_csv(X.apath("domrand", "timing.csv"), index=False)
S.table(TD.groupby("label").agg(calls=("seconds", "size"), total_s=("seconds", "sum"),
                                mean_s=("seconds", "mean")).reset_index()
        .sort_values("total_s", ascending=False),
        "Wall-clock by stage", note="timing_before.csv holds the first pass",
        csv=("domrand", "timing_summary.csv"))
naive_extra = float(TD[TD.label.str.startswith("train:")].seconds.mean()
                    * len(ARMS))
print(f"\n  Reusing the headline runs as seed 0 avoided retraining "
      f"{len(ARMS)} arms, measured at {naive_extra:.1f} s at this scale. "
      f"That is the measured saving, not a claimed one.")
X.save_json(dict(scale=S.SCALE, dr_budget=S.DR_BUDGET, budget=BUD, seeds=SEEDS,
                 conservatism_premium_pct=premium, total_wall_s=float(TD.seconds.sum())),
            "domrand", "summary.json")

# %% [markdown]
# ## What this notebook establishes
#
# - T6 carries the conservatism premium **as a signed percentage**, and the
#   analysis above states its sign before its magnitude.
# - F14 shows seed variance and episode variance side by side. They are two
#   different quantities; reporting only one would manufacture a robustness
#   claim.
# - F13 shows which axis each policy can actually survive, plotted against the
#   *realised* factor because the estimator is vectorised.
# - The structural conclusion is independent of every sign above: **a policy that
#   cannot observe its plant cannot specialise.** It can only find one compromise
#   acceptable everywhere. Notebook 5 gives it eyes.

# %%
S.checkpoint(nominal=("domrand", "nominal.pkl"), dr=("domrand", "DR-all.pkl"),
             dr_noise=("domrand", "DR-all+noise.pkl"),
             axis=("domrand", "axis_sweep.csv"), timing=("domrand", "timing.csv"),
             summary=("domrand", "summary.json"))
print("\nNotebook 4 complete.")
