# %% [markdown]
# # Notebook 6 — the comparison
#
# **Establishes.** The ledger, and answers to three questions.
#
# **Consumes.** every checkpoint produced so far.
#
# **Produces.** `common/ledger.csv`, `common/nb6_grid.csv`,
# `common/nb6_latency.csv`, figures F21–F24, tables T10–T11.
#
# | id | suite |
# |---|---|
# | S1 | nominal |
# | S2 | moderated in-distribution |
# | S3 | moderated out-of-distribution |
#
# **No PID.** It shares no design object with the others — different structure,
# nine hand gains, a different tuning budget — so its column would measure
# tuning effort rather than a controller class.

# %%
import _nbinit  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp

import adaptive_core_jax as A
import x500_core_jax as X
import study_prelude as S
import study_moderate as M
import viz as V

S.header("Notebook 6 - the comparison")
X.style()
FIG = V.figures_dir()

PATHS = ("circle", "fig8", "square")
SUITES = {"S1": S.nominal_spec(speed=(1.0, 1.5)),
          "S2": M.moderate(S.disturbed_spec(), wind=(0.0, 2.0)),
          "S3": M.moderate(S.ood_spec(), wind=(2.5, 4.0))}
M.check_moderate(S.disturbed_spec(), SUITES["S2"], 2.0)
M.check_moderate(S.ood_spec(), SUITES["S3"], 4.0)
SEEDS = dict(S1=1001, S2=1002, S3=1003)
X.save_json(dict(suites={k: {kk: list(vv) if isinstance(vv, tuple) else vv
                             for kk, vv in v.items()} for k, v in SUITES.items()},
                 seeds=SEEDS, paths=list(PATHS), n_eval=S.CFG["n_eval"],
                 T_eval=S.CFG["T_eval"], scale=S.SCALE),
            "common", "nb6_conditions.json")
print(M.disjointness(SUITES["S2"], SUITES["S3"]).to_string(index=False))
print("  -> disjoint on drag, lag and wind only; do NOT claim full disjointness (C-4).")

# %% [markdown]
# ## Controllers
#
# LLTC is rebuilt **explicitly** through `make_lltc_ctrl`. A generic loader keyed
# on `'actor'` raises `KeyError` on `lltc/model.pkl` and silently drops the row —
# the comparison would then quietly have one fewer controller.

# %%
S.section(1, "controllers", "assemble the row set, skipping nothing silently")

NIT = S.CFG["ilqr"]
CT = {}
CT["LQR"] = X.make_lqr_ctrl()
CT["NMPC N=1"] = X.make_nmpc_ctrl(N=1, n_iter=NIT)
CT["NMPC N=10"] = X.make_nmpc_ctrl(N=10, n_iter=NIT)
TUNED = {"LQR": 13, "NMPC N=1": 13, "NMPC N=10": 13}

lltc = X.load_ckpt("lltc", "model.pkl")
if lltc is not None:
    CT["LLTC N=1"] = M.make_lltc_ctrl(X, lltc, N=1, n_iter=NIT)
    TUNED["LLTC N=1"] = 13
    print(f"  LLTC rebuilt explicitly; its recorded fit R2 = "
          f"{lltc['fit']['R2']:+.3f}"
          f"{'  (gate FAILED -- its rows are a negative result)' if lltc['fit']['R2'] <= 0 else ''}")
else:
    print("  !! lltc/model.pkl missing -- LLTC row SKIPPED (run Notebook 2)")

acm = X.load_ckpt("acmpc", "model.pkl")
if acm is not None:
    CT["AC-MPC N=1"] = X.ctrl_from_actor(acm["actor"], dict(acm["cfg"], n_diff=1,
                                                            name="AC-MPC N=1"))
    TUNED["AC-MPC N=1"] = 0
else:
    print("  !! acmpc/model.pkl missing -- AC-MPC row SKIPPED (run Notebook 3)")

est = X.load_ckpt("acmpc_adaptive", "estimators", "all.pkl")
best_v = None
for v in ("C", "B", "A"):
    ckv = X.load_ckpt("acmpc_adaptive", "variants", f"{v}.pkl")
    if ckv is not None:
        best_v = (v, ckv)
        break
if best_v is not None and est is not None:
    v, ckv = best_v
    p_sel, sc_sel, Hsel = A.rebuild_rdp(est[est["__selected__"]])
    CT["Adaptive AC-MPC N=1"] = X.ctrl_from_actor(
        ckv["actor"], dict(ckv["cfg"], n_diff=1, name="Adaptive AC-MPC N=1"),
        dmod_fn=(lambda ev: ev.dmod()) if ckv.get("to_model") else None)
    TUNED["Adaptive AC-MPC N=1"] = 0
    print(f"  adaptive arm = variant {v}, estimator {est['__selected__']} (H={Hsel})")
else:
    print("  !! adaptive checkpoints missing -- Adaptive row SKIPPED (run Notebook 5)")

ORDER = list(CT)
assert "PID" not in ORDER, "PID must not appear in the main study comparison"
print(f"  controllers in the comparison: {ORDER}")

# %% [markdown]
# ## The grid

# %%
S.section(2, "grid", "3 suites x 3 paths x every controller",
          produces="common/nb6_grid.csv, F22, F24")

rows = []
for sid, spec in SUITES.items():
    for path in PATHS:
        for name in ORDER:
            need_rdp = name.startswith("Adaptive")
            if need_rdp:
                env = A.AdaptEnv(S.CFG["n_eval"], SEEDS[sid], 100000, spec,
                                 "central", 0.0, oracle=True, paths=(path,),
                                 H=Hsel)
                env.attach(p_sel, sc_sel)
            else:
                env = X.Env(S.CFG["n_eval"], SEEDS[sid], 100000, spec, (path,))
            st = X.stats(X.rollout_eval(env, CT[name], S.CFG["T_eval"] // 2,
                                        warmup=40))
            rows.append(dict(suite=sid, path=path, ctrl=name, **st))
GRID = pd.DataFrame(rows)
S.table(GRID[["suite", "path", "ctrl", "rmse", "maxerr", "tilt", "smooth", "sat",
              "crash", "bound_frac"]],
        "nb6 grid", note="saturation is reported beside RMSE everywhere: if the "
        "path stops mattering you are measuring the disturbance; bound_frac is "
        "the fraction of vehicle-steps that hit the 3 m respawn, above which "
        "rmse is a property of the bound and not of the controller",
        csv=("common", "nb6_grid.csv"))

# G9: the bound is a CEILING on rmse, so a diverging controller cannot report a
# large one.  Name the rows where that happened instead of letting the number
# read as a tracking error.
_bd = GRID[GRID.bound_frac > 0.005]
if len(_bd):
    print(f"\n  !! {len(_bd)} of {len(GRID)} cells respawned at the "
          f"{X.MAX_POS_ERR:g} m position bound on more than 0.5 % of steps.  In "
          f"those cells |e_p| is truncated at the bound, so `rmse` saturates "
          f"near {X.MAX_POS_ERR/np.sqrt(3):.2f} m however badly the controller "
          f"diverges -- it is NOT comparable with the rows that never respawned.")
    print(_bd.groupby("ctrl").bound_frac.max().sort_values(ascending=False)
          .to_string())
    print("     Re-read those rows with env.no_respawn = True (as Notebook 7 "
          "does) for the unclipped divergence.")

# %% [markdown]
# ## Latency — on a batch of ONE
#
# `ms_per_step / batch` is throughput. Only `solve_latency_ms` may be compared
# against the 20 ms period; mixing them produces an apparent 200x speed-up for
# the learned controllers.

# %%
S.section(3, "latency", "single-vehicle solve time", produces="common/nb6_latency.csv")
rows = []
for name in ORDER:
    if name.startswith("Adaptive"):
        env1 = A.AdaptEnv(1, 7, 100000, SUITES["S1"], "central", 0.0, oracle=True,
                          paths=("circle",), H=Hsel)
        env1.attach(p_sel, sc_sel)
    else:
        env1 = X.Env(1, 7, 100000, SUITES["S1"], ("circle",))
    lat = X.solve_latency_ms(CT[name], env1, T=30)
    rows.append(dict(ctrl=name, **lat, tuned=TUNED[name],
                     admissible=bool(lat["p95"] < 20.0)))
LAT = pd.DataFrame(rows)
S.table(LAT, "nb6 latency, batch of ONE",
        note="throughput (ms_per_step/batch) never appears in this table",
        csv=("common", "nb6_latency.csv"))

# %% [markdown]
# ## T10 — the ledger

# %%
S.section(4, "ledger", "the headline table", produces="common/ledger.csv, T10")
piv = GRID.pivot_table(index="ctrl", columns="suite", values="rmse")
T10 = piv.reindex(ORDER).reset_index().merge(LAT, on="ctrl")
T10 = T10.rename(columns={"S1": "rmse_S1", "S2": "rmse_S2", "S3": "rmse_S3",
                          "median": "ms_median", "p95": "ms_p95"})
T10["scale"] = S.SCALE
T10 = T10[["ctrl", "rmse_S1", "rmse_S2", "rmse_S3", "ms_median", "ms_p95",
           "tuned", "admissible", "scale"]]
S.table(T10, "T10  THE LEDGER",
        note="ms columns are single-vehicle solve latency; tuned is the count of "
             "hand-chosen stage weights",
        csv=("common", "ledger.csv"))

sat_pin = GRID[GRID.sat > 0.4]
if len(sat_pin):
    print(f"\n  GUARD: {len(sat_pin)} of {len(GRID)} cells exceed 40 % collective "
          f"saturation.  In those cells the path has stopped mattering and the "
          f"number is a property of the disturbance, not of the controller:")
    print(sat_pin.groupby(["suite", "ctrl"]).sat.max().to_string())

# %% [markdown]
# ## T11 — degradation, and the three questions

# %%
S.section(5, "degradation and the three questions", "answer them in order, with numbers",
          produces="common/nb6_degradation.csv, T11")

T11 = T10[["ctrl", "rmse_S1", "rmse_S2", "rmse_S3"]].copy()
T11["S2/S1"] = T11.rmse_S2 / T11.rmse_S1.clip(lower=1e-9)
T11["S3/S1"] = T11.rmse_S3 / T11.rmse_S1.clip(lower=1e-9)
T11["rank_S1"] = T11.rmse_S1.rank().astype(int)
T11["rank_S3"] = T11.rmse_S3.rank().astype(int)
T11["rank_change"] = T11.rank_S1 - T11.rank_S3
S.table(T11, "T11  degradation factors and rank change",
        note="moderated-S3 degradation factors are NOT comparable with "
             "pre-moderation ones -- the suite is a different, narrower object",
        csv=("common", "nb6_degradation.csv"))

print("\n  QUESTION 1 -- does optimisation in the loop earn its compute?")
print("    LQR vs NMPC N=1: same model, same weights, same feed-forward "
      "information, so the difference is exactly what the online solve buys.")
for path in PATHS:
    d = GRID[(GRID.suite == "S1") & (GRID.path == path)].set_index("ctrl")
    if "LQR" in d.index and "NMPC N=1" in d.index:
        l, n = float(d.loc["LQR", "rmse"]), float(d.loc["NMPC N=1", "rmse"])
        print(f"      {path:7s}: LQR {l:.4f} -> NMPC N=1 {n:.4f} m "
              f"({100*(l-n)/max(l,1e-9):+.1f}% for "
              f"{float(LAT[LAT.ctrl=='NMPC N=1'].ms_median.iloc[0] if 'ms_median' in LAT else LAT[LAT.ctrl=='NMPC N=1']['median'].iloc[0]):.2f} ms)")
print("    If it is small on smooth paths and large on the superellipse, that is "
      "the honest answer.")

print("\n  QUESTION 2 -- does the learned cost recover long-horizon behaviour at "
      "one-step cost?")
print("    Read RMSE and p95 latency together; this is the only place the claim "
      "is tested against both ends at once.")
for name in ORDER:
    r = T10[T10.ctrl == name].iloc[0]
    print(f"      {name:22s} S1 {r.rmse_S1:.4f} m   p95 {r.ms_p95:6.2f} ms   "
          f"tuned {int(r.tuned):2d}   {'admissible' if r.admissible else 'OVER BUDGET'}")

print("\n  QUESTION 3 -- what survives leaving the training distribution?")
for _, r in T11.iterrows():
    print(f"      {r.ctrl:22s} S3/S1 = {r['S3/S1']:6.2f}x   rank {r.rank_S1} -> "
          f"{r.rank_S3} ({r.rank_change:+d})")
tuned_not_designed = T11[(T11.rank_S1 <= 2) & (T11.rank_S3 >= len(T11) - 1)]
if len(tuned_not_designed):
    print(f"    A controller that wins on S1 and loses on S3 has been tuned, not "
          f"designed: {list(tuned_not_designed.ctrl)}")
else:
    print("    No controller both wins on S1 and loses on S3 in this run.")
S.scale_guard(GRID[GRID.suite == "S1"])

# %% [markdown]
# ## Figures

# %%
fig, ax = plt.subplots(figsize=(6.4, 4.4))
mk = ["o", "s", "^", "v", "D", "P", "X"]
for i, (_, r) in enumerate(T10.iterrows()):
    ax.scatter(r.ms_p95, r.rmse_S1, s=60 + 25 * int(r.tuned), marker=mk[i % len(mk)],
               label=f"{r.ctrl} (tuned {int(r.tuned)})", zorder=3)
ax.axvline(20.0, color="C3", lw=1.6)
ax.text(20.6, ax.get_ylim()[1] * 0.95, "20 ms admissibility", color="C3",
        fontsize=7, rotation=90, va="top")
ax.set_xscale("log"); ax.set_xlabel("single-vehicle p95 solve latency [ms] (log)")
ax.set_ylabel("position RMSE on S1 [m]"); ax.legend(fontsize=6.5)
ax.set_title("F21  HEADLINE: accuracy vs latency; marker size = tuned parameters")
fig.savefig(f"{FIG}/F21_headline.png", bbox_inches="tight")
T10.to_csv(f"{FIG}/F21_headline.csv", index=False)

fig, ax = plt.subplots(figsize=(7.6, 3.6))
xs = np.arange(len(ORDER))
w = 0.26
for j, sid in enumerate(SUITES):
    v = [float(piv.loc[c, sid]) for c in ORDER]
    ax.bar(xs + (j - 1) * w, v, w, label=sid, hatch=["", "//", ".."][j],
           edgecolor="k", lw=0.5)
ax.set_xticks(xs); ax.set_xticklabels(ORDER, rotation=18, ha="right", fontsize=7)
ax.set_ylabel("position RMSE [m]"); ax.legend(fontsize=7)
ax.set_title("F22  RMSE by condition x controller")
fig.savefig(f"{FIG}/F22_by_condition.png", bbox_inches="tight")
piv.to_csv(f"{FIG}/F22_by_condition.csv")

# F23 radar
AXES = ["accuracy", "robustness", "smoothness", "effort", "latency", "tuning"]
raw = pd.DataFrame({
    "accuracy": 1.0 / T10.rmse_S1.clip(lower=1e-9),
    "robustness": 1.0 / T11["S3/S1"].clip(lower=1e-9).values,
    "smoothness": 1.0 / GRID[GRID.suite == "S1"].groupby("ctrl").smooth.mean()
                  .reindex(ORDER).clip(lower=1e-12).values,
    "effort": 1.0 / GRID[GRID.suite == "S1"].groupby("ctrl").effort.mean()
              .reindex(ORDER).clip(lower=1e-12).values,
    "latency": 1.0 / T10.ms_p95.clip(lower=1e-9),
    "tuning": 1.0 / (T10.tuned.values + 1.0)}, index=ORDER)
norm = raw / raw.max()
ang = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False).tolist()
ang += ang[:1]
fig = plt.figure(figsize=(5.4, 5.0))
ax = fig.add_subplot(111, polar=True)
for i, c in enumerate(ORDER):
    v = norm.loc[c, AXES].tolist(); v += v[:1]
    ax.plot(ang, v, lw=1.4, marker=mk[i % len(mk)], ms=4, label=c)
    ax.fill(ang, v, alpha=0.06)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(AXES, fontsize=7)
ax.set_yticklabels([]); ax.legend(fontsize=6, loc="upper right",
                                  bbox_to_anchor=(1.3, 1.1))
ax.set_title("F23  normalised per axis (higher is better)", fontsize=9)
fig.savefig(f"{FIG}/F23_radar.png", bbox_inches="tight")
norm.to_csv(f"{FIG}/F23_radar.csv")

fig, ax = plt.subplots(figsize=(7.2, 3.4))
sp = GRID.pivot_table(index="ctrl", columns="suite", values="sat").reindex(ORDER)
for j, sid in enumerate(SUITES):
    ax.bar(xs + (j - 1) * w, sp[sid].values, w, label=sid,
           hatch=["", "//", ".."][j], edgecolor="k", lw=0.5)
ax.axhline(0.4, color="C3", lw=1.4)
ax.text(0, 0.41, "40 %: the path has stopped mattering", color="C3", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(ORDER, rotation=18, ha="right", fontsize=7)
ax.set_ylabel("collective saturation fraction"); ax.legend(fontsize=7)
ax.set_title("F24  the guard figure")
fig.savefig(f"{FIG}/F24_saturation_guard.png", bbox_inches="tight")
sp.to_csv(f"{FIG}/F24_saturation_guard.csv")

# %% [markdown]
# ## What this notebook establishes
#
# The three questions are answered above, in order, with numbers from this run.
# Two cautions travel with them:
#
# - **Moderated-S3 degradation factors are not comparable with pre-moderation
#   ones.** The moderated suite is a narrower object, deliberately, and a factor
#   computed against it means something different.
# - **Saturation is reported beside every RMSE.** Where F24 shows a bar above
#   40 %, the corresponding RMSE is a property of the disturbance, not of the
#   controller, and the ledger row should be read with that in mind.

# %%
S.checkpoint(ledger=("common", "ledger.csv"), grid=("common", "nb6_grid.csv"),
             latency=("common", "nb6_latency.csv"),
             conditions=("common", "nb6_conditions.json"))
print("\nNotebook 6 complete.")
