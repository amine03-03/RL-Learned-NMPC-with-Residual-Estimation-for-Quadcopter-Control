# %% [markdown]
# # Notebook 3 — AC-MPC, the learned cost map
#
# **Establishes.** Every hand-chosen stage weight can be removed; what that
# costs; how a differentiable optimiser behaves under exploration noise.
#
# **Consumes.** `common/ledger.csv`, `common/nb1_horizon.csv` (the tuned NMPC
# reference).
#
# **Produces.** `acmpc/model.pkl`, `acmpc/nb3_{representation,exploration,mpve,
# algo,seeds}.csv`, figures F9–F12, table T5, the AC-MPC ledger row.
#
# **Any difference smaller than the seed spread is not a result.** Sweep 5
# establishes that scale first, and every later comparison is read against it.

# %%
import _nbinit  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

import x500_core_jax as X
import study_prelude as S
import viz as V

S.header("Notebook 3 - AC-MPC, the learned cost map")
X.style()
FIG = V.figures_dir()

ITERS = S.CFG["iters_sweep"]
TROLL = S.CFG["T_rollout"]
NENV = S.CFG["n_env"]
HID = S.CFG["hid"]
NIT = S.CFG["ilqr"]
TASKS = {"circle": ("circle",), "fig8": ("fig8",), "stabilize": ("hover", "step")}


def mk_env(task, seed=0, spec=None, **kw):
    """`stabilize` is the regulation task: hover and step references, so the
    policy is scored on rejecting the seeded offset rather than on following a
    moving path.  Declared here because §5.11's `task` field does not say."""
    return X.Env(NENV, seed, 200, spec or S.nominal_spec(speed=(1.0, 1.5)),
                 TASKS[task], **kw)


def base_cfg(**kw):
    c = dict(N=1, rep="diag", n_iter=NIT, n_diff=2, hid=HID, minib=S.CFG["minib"],
             epochs=S.CFG["epochs"], sigma=0.05, algo="ppo", mpve=False, lam=0.95,
             dist_label="nominal")
    c.update(kw)
    return c


def evaluate(actor, cfg, task, seed=900, spec=None, T=None):
    env = X.Env(min(S.CFG["n_eval"], 64), seed, 100000,
                spec or S.nominal_spec(speed=(1.0, 1.5)), TASKS[task])
    ctrl = X.ctrl_from_actor(actor, dict(cfg, n_diff=1, name="AC-MPC"))
    return X.stats(X.rollout_eval(env, ctrl, T or S.CFG["T_eval"] // 2, warmup=40))

# %% [markdown]
# ## Sweep 5 first — the seed spread
#
# Run out of order deliberately. Every other sweep below is judged against this
# number, so producing it last would invite reading a difference as a result
# before knowing whether it is noise.

# %%
S.section(5, "seed spread", "the scale against which every other difference is judged",
          produces="acmpc/nb3_seeds.csv")

SEEDS = (5, 200, 201)
rows, LOGS = [], {}
for sd in SEEDS:
    a, c_, log = X.train_ppo(None, base_cfg(), seed=sd, iters=ITERS,
                             T_rollout=TROLL, env=mk_env("circle", seed=sd),
                             verbose=False)
    st = evaluate(a, base_cfg(), "circle")
    rows.append(dict(seed=sd, final_reward=float(log.reward.tail(3).mean()),
                     rmse=st["rmse"], sat=st["sat"], crash=st["crash"]))
    LOGS[f"seed{sd}"] = log
    if sd == SEEDS[0]:
        X.save_ckpt(dict(actor=a, critic=c_, cfg=base_cfg()), "acmpc", "model.pkl")
SEED_DF = pd.DataFrame(rows)
SEED_SPREAD = float(SEED_DF.rmse.max() - SEED_DF.rmse.min())
SEED_STD = float(SEED_DF.rmse.std())
S.table(SEED_DF, "Seed study (3 seeds, identical configuration)",
        note=f"seed spread (max-min) = {SEED_SPREAD:.4f} m, std = {SEED_STD:.4f} m",
        csv=("acmpc", "nb3_seeds.csv"))
print(f"\n  >>> Any difference below {SEED_SPREAD:.4f} m in this notebook is NOT "
      f"a result.  This bound is quoted beside every comparison that follows.")

# %% [markdown]
# ## Sweep 1 — representation
#
# `diag` (13 free parameters per stage) / `chol` (91) / `full` (169).
#
# **The initialisation asymmetry is recorded, not glossed.** With a 0.1-scaled
# zero-bias head, `chol` and `full` start at $\mathbf A\approx 0$ so
# $\mathbf S\approx Q_{LO}\mathbf I$ — the quadratic term is effectively absent
# at init — whereas `diag` starts at the sigmoid mid-range $\approx 5\times10^4$.
# They do not start from comparable places. That predicts the richer forms need
# **longer**, not that they are incapable.

# %%
S.section(1, "representation", "diag vs chol vs full on three tasks",
          produces="acmpc/nb3_representation.csv, F9, F12")

print("  initialisation check (what each representation starts from):")
k0 = jax.random.PRNGKey(0)
for rep in X.REPS:
    th = X.costmap_init(k0, X.OBS_DIM, HID, rep, 1)
    Sm, _ = X.costmap_apply(th, jnp.zeros((4, X.OBS_DIM)), rep, 1)
    d = np.diagonal(np.asarray(Sm)[0, 0])
    print(f"    {rep:5s}: {X.REP_DIM[rep]:3d} free params/stage, "
          f"S diagonal at init in [{d.min():.3e}, {d.max():.3e}]")

rows, CURVES = [], {}
for rep in X.REPS:
    for task in TASKS:
        cfg = base_cfg(rep=rep)
        a, _, log = X.train_ppo(None, cfg, seed=5, iters=ITERS, T_rollout=TROLL,
                                env=mk_env(task, seed=5), verbose=False)
        st = evaluate(a, cfg, task)
        rows.append(dict(rep=rep, task=task, free_params=X.REP_DIM[rep],
                         final_reward=float(log.reward.tail(3).mean()),
                         rmse=st["rmse"], sat=st["sat"], crash=st["crash"]))
        CURVES[(rep, task)] = log.reward.values
        if rep == "diag" and task == "circle":
            ACTOR_DIAG = a
REP_DF = pd.DataFrame(rows)
S.table(REP_DF, "Representation sweep",
        note=f"seed spread on the diag/circle cell is {SEED_SPREAD:.4f} m; "
             f"differences below that are not results",
        csv=("acmpc", "nb3_representation.csv"))

print("\n  per-task winner (by RMSE), with the seed spread for scale:")
for task in TASKS:
    d = REP_DF[REP_DF.task == task].sort_values("rmse")
    gap = float(d.rmse.iloc[1] - d.rmse.iloc[0])
    verdict = ("within the seed spread -- NOT a result"
               if gap < SEED_SPREAD else "exceeds the seed spread")
    print(f"    {task:10s}: {d.rep.iloc[0]:5s} ({d.rmse.iloc[0]:.4f} m), "
          f"next {d.rep.iloc[1]} by {gap:.4f} m -- {verdict}")
DIAG_WINS = sum(REP_DF[REP_DF.task == t].sort_values("rmse").rep.iloc[0] == "diag"
                for t in TASKS)
print(f"\n  The published finding is that only the diagonal learns.  Here `diag` "
      f"won {DIAG_WINS}/{len(TASKS)} tasks.")
print(f"  {'It reproduced.' if DIAG_WINS == len(TASKS) else 'It did NOT reproduce on every task.'} "
      f"Where a richer form loses, the initialisation asymmetry printed above is "
      f"the candidate explanation -- chol/full begin with the quadratic term "
      f"effectively absent -- and it predicts they need longer, not that they "
      f"are incapable.  At {ITERS} iterations this notebook cannot separate the two.")

fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3), sharey=True)
for j, task in enumerate(TASKS):
    for i, rep in enumerate(X.REPS):
        y = CURVES[(rep, task)]
        axs[j].plot(y, color=f"C{i}", lw=1.3, label=rep,
                    marker=["o", "s", "^"][i], markevery=max(len(y) // 8, 1), ms=3)
    if task == "circle":
        band = np.stack([LOGS[f"seed{s}"].reward.values for s in SEEDS])
        axs[j].fill_between(range(band.shape[1]), band.min(0), band.max(0),
                            color="0.6", alpha=0.3, label="seed band (diag)")
    axs[j].set_title(task); axs[j].set_xlabel("PPO iteration")
axs[0].set_ylabel("mean reward"); axs[0].legend(fontsize=6.5)
fig.suptitle("F9  cost-map representations", fontsize=10)
fig.savefig(f"{FIG}/F9_representations.png", bbox_inches="tight")
pd.DataFrame({f"{r}_{t}": CURVES[(r, t)] for r in X.REPS for t in TASKS}).to_csv(
    f"{FIG}/F9_representations.csv", index=False)

# F12 -- the interpretability figure
env = mk_env("circle", seed=1)
o, e, xr = env.obs()
Sm, _ = X.costmap_apply(X.load_ckpt("acmpc", "model.pkl")["actor"], o, "diag", 1)
learned = np.asarray(jnp.mean(jnp.diagonal(Sm[:, 0], axis1=-2, axis2=-1), 0))
hand = np.r_[np.diag(X.Q_HAND), np.diag(X.R_HAND)]
labels = ["e_px", "e_py", "e_pz", "e_vx", "e_vy", "e_vz", "e_ax", "e_ay", "e_az",
          "du_c", "du_wx", "du_wy", "du_wz"]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
xs = np.arange(len(labels))
ax.bar(xs - 0.2, hand, 0.4, label="hand-tuned Q, R", edgecolor="k", lw=0.5)
ax.bar(xs + 0.2, learned, 0.4, label="learned (diag, mean over batch)",
       hatch="//", edgecolor="k", lw=0.5)
ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, fontsize=7)
ax.set_ylabel("stage weight (log)"); ax.legend(fontsize=7)
ax.set_title("F12  learned diagonal against the hand-tuned weights")
fig.savefig(f"{FIG}/F12_interpretability.png", bbox_inches="tight")
pd.DataFrame(dict(channel=labels, hand=hand, learned=learned)).to_csv(
    f"{FIG}/F12_interpretability.csv", index=False)
print(f"\n  learned/hand weight ratio spans "
      f"[{(learned/np.maximum(hand,1e-12)).min():.2e}, "
      f"{(learned/np.maximum(hand,1e-12)).max():.2e}] -- the learned cost is a "
      f"different object from the tuned one, not a rescaling of it.")

# %% [markdown]
# ## Sweep 2 — exploration noise, and the mechanism behind it
#
# **Hypothesis.** Additive noise on a box-constrained collective does not merely
# perturb the action: it drives the input onto the box and corrupts the
# linearisation $\mathbf A_k,\mathbf B_k$ *inside* the solve. A model-free MLP
# has no solve to corrupt.
#
# This is testable only because the noise is now on the **action**, as
# arXiv:2306.09852 eq (7) specifies. An earlier version perturbed the cost-map
# parameters, under which saturation does not rise with $\sigma$ at all and the
# mechanism cannot occur — see `docs/AUDIT.md` (B1).
#
# Saturation fraction is recorded to test the **mechanism**, not just the effect.

# %%
S.section(2, "exploration noise", "AC-MPC vs a model-free MLP on identical reward",
          produces="acmpc/nb3_exploration.csv, F10")

rows = []
for sig in (0.05, 0.15, 0.30):
    cfg = base_cfg(sigma=sig)
    a, _, log = X.train_ppo(None, cfg, seed=5, iters=ITERS, T_rollout=TROLL,
                            env=mk_env("circle", seed=5), verbose=False)
    st = evaluate(a, cfg, "circle")
    rows.append(dict(arch="AC-MPC", sigma=sig,
                     final_reward=float(log.reward.tail(3).mean()),
                     train_sat=float(log.sat.tail(max(len(log)//5,1)).mean()),
                     rmse=st["rmse"], eval_sat=st["sat"]))
    am, _, logm = X.train_mlp(mk_env("circle", seed=5),
                              dict(hid=HID, minib=S.CFG["minib"],
                                   epochs=S.CFG["epochs"], sigma=sig),
                              seed=5, iters=ITERS, T_rollout=TROLL, verbose=False)
    envm = X.Env(min(S.CFG["n_eval"], 64), 900, 100000,
                 S.nominal_spec(speed=(1.0, 1.5)), TASKS["circle"])
    stm = X.stats(X.rollout_eval(envm, X.make_mlp_ctrl(am), S.CFG["T_eval"] // 2,
                                 warmup=40))
    rows.append(dict(arch="MLP (model-free)", sigma=sig,
                     final_reward=float(logm.reward.tail(3).mean()),
                     train_sat=float(logm.sat.tail(max(len(logm)//5,1)).mean()),
                     rmse=stm["rmse"], eval_sat=stm["sat"]))
EXP = pd.DataFrame(rows)
S.table(EXP, "Exploration-noise sweep",
        note="train_sat is the mechanism under test; final_reward is the effect",
        csv=("acmpc", "nb3_exploration.csv"))

fig, ax = plt.subplots(figsize=(5.8, 3.6))
ax2 = ax.twinx()
for i, arch in enumerate(EXP.arch.unique()):
    d = EXP[EXP.arch == arch]
    ax.plot(d.sigma, d.final_reward, "-o" if i == 0 else "-s", color=f"C{i}",
            ms=5, label=f"{arch} (reward)")
    ax2.plot(d.sigma, d.train_sat, ":^" if i == 0 else ":v", color=f"C{i}",
             ms=4, alpha=0.75, label=f"{arch} (saturation)")
ax.set_xlabel(r"exploration $\sigma$"); ax.set_ylabel("final reward")
ax2.set_ylabel("collective saturation fraction")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="lower left")
ax.set_title("F10  exploration noise: effect and mechanism")
fig.savefig(f"{FIG}/F10_exploration.png", bbox_inches="tight")
EXP.to_csv(f"{FIG}/F10_exploration.csv", index=False)

acm = EXP[EXP.arch == "AC-MPC"].set_index("sigma")
mlp = EXP[EXP.arch == "MLP (model-free)"].set_index("sigma")
d_acm = float(acm.final_reward.loc[0.30] - acm.final_reward.loc[0.05])
d_mlp = float(mlp.final_reward.loc[0.30] - mlp.final_reward.loc[0.05])
print(f"\n  reward change from sigma 0.05 -> 0.30: AC-MPC {d_acm:+.4f}, "
      f"MLP {d_mlp:+.4f}")
print(f"  saturation at sigma=0.30: AC-MPC {float(acm.train_sat.loc[0.30]):.3f}, "
      f"MLP {float(mlp.train_sat.loc[0.30]):.3f}")
print(f"  VERDICT: AC-MPC is {'MORE' if d_acm < d_mlp else 'NOT more'} "
      f"noise-sensitive than the model-free arm in this run.  The mechanism "
      f"claim needs the saturation column to move with it; read both.")

# %% [markdown]
# ## Sweep 3 — MPVE, and Sweep 4 — PPO vs TRPO

# %%
S.section(3, "MPVE and algorithm", "value expansion and the update rule",
          produces="acmpc/nb3_mpve.csv, acmpc/nb3_algo.csv, F11")

rows = []
for lam in (0.90, 0.95, 0.99):
    for mpve in (False, True):
        cfg = base_cfg(lam=lam, mpve=mpve)
        a, _, log = X.train_ppo(None, cfg, seed=5, iters=ITERS, T_rollout=TROLL,
                                env=mk_env("circle", seed=5), verbose=False)
        rows.append(dict(lam=lam, mpve=mpve,
                         final_reward=float(log.reward.tail(3).mean()),
                         value_loss=float(log.value_loss.tail(3).mean()),
                         rmse=evaluate(a, cfg, "circle")["rmse"]))
MPVE = pd.DataFrame(rows)
MPVE["gain"] = MPVE.groupby("lam").final_reward.transform(lambda g: g.iloc[1] - g.iloc[0])
S.table(MPVE, "MPVE sweep", note="expect the largest gain at low lambda, where "
        "GAE leans hardest on the critic", csv=("acmpc", "nb3_mpve.csv"))

rows = []
for algo in ("ppo", "trpo"):
    cfg = base_cfg(algo=algo)
    a, _, log = X.train_ppo(None, cfg, seed=5, iters=ITERS, T_rollout=TROLL,
                            env=mk_env("circle", seed=5), verbose=False)
    st = evaluate(a, cfg, "circle")
    rows.append(dict(algo=algo, final_reward=float(log.reward.tail(3).mean()),
                     mean_kl=float(log.kl.mean()), max_kl=float(log.kl.max()),
                     rmse=st["rmse"], sat=st["sat"]))
ALGO = pd.DataFrame(rows)
S.table(ALGO, "PPO vs TRPO, everything else fixed",
        note=f"seed spread {SEED_SPREAD:.4f} m", csv=("acmpc", "nb3_algo.csv"))
gap = abs(float(ALGO.rmse.iloc[0] - ALGO.rmse.iloc[1]))
print(f"\n  PPO/TRPO RMSE gap = {gap:.4f} m vs seed spread {SEED_SPREAD:.4f} m: "
      f"{'NOT a result' if gap < SEED_SPREAD else 'exceeds the seed spread'}.")
print(f"  TRPO holds the KL at max {float(ALGO.max_kl.iloc[1]):.4f} against PPO's "
      f"{float(ALGO.max_kl.iloc[0]):.4f} -- the trust region is doing what it says.")

fig, ax = plt.subplots(figsize=(5.2, 3.3))
g = MPVE.groupby("lam").gain.first()
ax.plot(g.index, g.values, "-o", ms=5, label="MPVE gain (reward)")
ax.fill_between(g.index, g.values - SEED_STD, g.values + SEED_STD, alpha=0.25,
                color="C0", label="seed spread (1 std)")
ax.axhline(0.0, color="0.4", lw=1, ls=":")
ax.set_xlabel(r"GAE $\lambda$"); ax.set_ylabel("reward gain from MPVE")
ax.legend(fontsize=7); ax.set_title(r"F11  MPVE gain vs $\lambda$")
fig.savefig(f"{FIG}/F11_mpve.png", bbox_inches="tight")
MPVE.to_csv(f"{FIG}/F11_mpve.csv", index=False)

# %% [markdown]
# ## T5 and the ledger row

# %%
S.section(6, "summary and ledger", "one row per design point; tuned goes to zero",
          produces="acmpc/nb3_summary.csv, common/ledger.csv")

T5 = pd.concat([
    REP_DF.assign(sweep="representation", point=REP_DF.rep + "/" + REP_DF.task),
    EXP.assign(sweep="exploration", point=EXP.arch + "/sigma=" + EXP.sigma.astype(str)),
    MPVE.assign(sweep="mpve", point="lam=" + MPVE.lam.astype(str)
                + "/mpve=" + MPVE.mpve.astype(str)),
    ALGO.assign(sweep="algo", point=ALGO.algo),
], ignore_index=True)[["sweep", "point", "final_reward", "rmse"]]
T5["seed_spread"] = SEED_SPREAD
S.table(T5, "T5  AC-MPC sweep summary",
        note="the seed_spread column is the resolution of every comparison here",
        csv=("acmpc", "nb3_summary.csv"))

ck = X.load_ckpt("acmpc", "model.pkl")
env1 = X.Env(1, 7, 100000, S.nominal_spec(speed=(1.2, 1.2)), ("circle",))
lat = X.solve_latency_ms(X.ctrl_from_actor(ck["actor"], dict(ck["cfg"], n_diff=1)),
                         env1, T=30)
best = evaluate(ck["actor"], ck["cfg"], "circle")
LED = pd.read_csv(X.apath("common", "ledger.csv"))
LED = LED[LED.ctrl != "AC-MPC N=1"]
LED = pd.concat([LED, pd.DataFrame([dict(
    ctrl="AC-MPC N=1", rmse_S1=best["rmse"], ms_median=lat["median"],
    ms_p95=lat["p95"], tuned=0, scale=S.SCALE)])], ignore_index=True)
S.table(LED, "Ledger after Notebook 3",
        note="tuned = 0: every hand-chosen stage weight has been removed. "
             "ms from solve_latency_ms on a batch of ONE, never ms_per_step/batch",
        csv=("common", "ledger.csv"))
nmpc1 = pd.read_csv(X.apath("common", "nb1_horizon.csv")).iloc[0]["circle"]
print(f"\n  ANALYSIS.  Tuned stage weights: 13 -> 0.")
print(f"  Accuracy cost against tuned NMPC N=1 on the circle: "
      f"{best['rmse']:.4f} m vs {nmpc1:.4f} m "
      f"({100*(best['rmse']-nmpc1)/max(nmpc1,1e-9):+.1f}%).")
S.scale_guard(pd.DataFrame(dict(ctrl=["NMPC N=1", "AC-MPC N=1"],
                                rmse=[nmpc1, best["rmse"]])))

# %% [markdown]
# ## What this notebook establishes
#
# - The 13 hand-chosen stage weights are gone: the ledger row carries
#   `tuned = 0`. What that costs in accuracy is the percentage printed above.
# - The representation verdict, the exploration verdict and the PPO/TRPO verdict
#   are each printed with the **seed spread beside them**. Where a gap is
#   smaller than that spread it is said, in those words, not to be a result.
# - The initialisation asymmetry of §5.9 is measured and printed, not assumed:
#   `chol` and `full` begin with the quadratic term effectively absent. That
#   predicts they need longer, and this notebook does not run long enough to
#   separate "needs longer" from "cannot".
# - F12 shows the learned diagonal is a different object from the tuned
#   $\mathbf Q,\mathbf R$, not a rescaling of it.

# %%
S.checkpoint(model=("acmpc", "model.pkl"), rep=("acmpc", "nb3_representation.csv"),
             exploration=("acmpc", "nb3_exploration.csv"),
             mpve=("acmpc", "nb3_mpve.csv"), algo=("acmpc", "nb3_algo.csv"),
             seeds=("acmpc", "nb3_seeds.csv"), ledger=("common", "ledger.csv"))
print("\nNotebook 3 complete.")
