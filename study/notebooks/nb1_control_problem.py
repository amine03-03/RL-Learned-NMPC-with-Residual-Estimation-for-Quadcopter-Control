# %% [markdown]
# # Notebook 1 — the control problem and the cost of an exact solution
#
# **Establishes.** The reference must be flyable; accuracy improves with horizon
# and costs compute; somebody has to choose $\mathbf Q,\mathbf R$ — and that
# choice moves the result a lot. That last measurement motivates the entire
# study.
#
# **Consumes.** Nothing. This is the root of the artefact tree.
#
# **Produces.** `common/path_feasibility.csv`, `lltc/nb1_classical.csv`,
# `common/nb1_horizon.csv`, `common/nb1_noise.csv`, `common/nb1_qr_sweep.csv`,
# `common/ledger.csv` (first row), figures F1–F5, table T1–T2.
#
# Every claim in the closing markdown is computed and printed by a cell. Where a
# table and this prose disagree, the prose is wrong.

# %%
import _nbinit  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp

import x500_core_jax as X
import study_prelude as S
import viz as V

S.header("Notebook 1 - the control problem")
X.style()
FIG = V.figures_dir()
RNG = np.random.default_rng(0)

# %% [markdown]
# ## Step 1 — physics self-test
#
# Assert §2's constants and tests T-1…T-5. **Refuse to continue on failure**: a
# wrong constant here is not a small error, it is a different aircraft.

# %%
S.section(1, "physics self-test", "assert §2 and T-1..T-5 before anything else",
          produces="common/nb1_selftest.csv")

rows = [
    ("m [kg]", X.M_TOT, 2.064308),
    ("cg0_z [m]", float(X.CG_NOM[2]), 0.001869131),
    ("J_xx [kg m^2]", float(X.J_NOM[0, 0]), 0.023830955),
    ("J_zz [kg m^2]", float(X.J_NOM[2, 2]), 0.043893959),
    ("f_max [N]", X.F_MAX, 8.54858),
    ("T_max [N]", X.T_MAX, 34.19432),
    ("T/W [-]", X.TW, 1.689122),
    ("u_hover [-]", X.U_HOVER, 0.769431),
    ("a_lat_max [m/s^2]", X.A_LAT_MAX, 13.349711),
    ("alpha*a_lat [m/s^2]", X.P.alpha_feas * X.A_LAT_MAX, 8.009827),
    ("cond(M) [-]", float(np.linalg.cond(X.M_NOM)), 62.5),
    ("d a_z/d c [1/s^2]", X.DAZ_DC_HOVER, 25.490538),
    ("kappa_a fig8 [-]", X.KAPPA_A[3], 2.125),
    ("kappa_a square [-]", X.KAPPA_A[5], 9.077877),
]
par = X.make_par(1)
s_h, u_h = X.hover_state(1, par=par), X.hover_u(1, par=par)
h = 1e-6
fd = float((X.fc(X.plant_to_ctrl(s_h), u_h.at[:, 0].add(h))[0, 5]
            - X.fc(X.plant_to_ctrl(s_h), u_h.at[:, 0].add(-h))[0, 5]) / (2 * h))
rows.append(("T-5 d a_z/d c (finite diff)", fd, 25.490538))
rows.append(("T-8 |residual| on nominal plant",
             float(jnp.abs(X.true_disturbance(s_h, u_h, par)).max()), 0.0))
rows.append(("T-8 |wrench| on nominal plant",
             float(jnp.abs(X.external_wrench(s_h, u_h, par, X.fp(s_h, u_h, par))).max()),
             0.0))

ST = pd.DataFrame(rows, columns=["symbol", "computed", "expected"])
ST["abs_delta"] = (ST.computed - ST.expected).abs()
ST["ok"] = ST.abs_delta <= 1e-6 * ST.expected.abs().clip(lower=1.0)
S.table(ST, "Physics self-test (T-1 .. T-5, T-8)",
        note="every value is derived from the SDF constants, none is pasted",
        csv=("common", "nb1_selftest.csv"))
if not ST.ok.all():
    raise SystemExit("PHYSICS SELF-TEST FAILED -- refusing to continue:\n"
                     + ST[~ST.ok].to_string(index=False))
print("\n  self-test passed; the hover linearisation is also asserted at import "
      "(C-1/C-2, see docs/CORRECTIONS.md).")

# %% [markdown]
# ## Step 2 — reference feasibility
#
# For each kind and $R\in[0.5,2.0]$, the peak $\|\mathbf a_{\rm ref}\|$ before
# and after the cap (4.6). Without the cap, a benchmark measures the reference
# rather than the controller.

# %%
S.section(2, "reference feasibility", "which references could not be flown by ANY controller",
          produces="common/path_feasibility.csv, F1")

SPD = 1.5
rowsF = []
for kind in X.PATHS_ALL:
    for R in np.linspace(0.5, 2.0, 16):
        w_un = SPD / max(R, 0.3) if kind not in ("hover", "step") else 0.0
        w_cap = float(X.path_omega(kind, R, SPD))
        pv_un, pa_un = X.path_demand(kind, R, w_un)
        pv, pa = X.path_demand(kind, R, w_cap)
        rowsF.append(dict(kind=kind, R=R, omega=w_cap,
                          period=(2 * np.pi / w_cap) if w_cap > 0 else np.nan,
                          peak_v=pv, peak_a=pa, peak_a_uncapped=pa_un,
                          capped=bool(w_cap < w_un - 1e-9)))
PF = pd.DataFrame(rowsF)
S.table(PF.groupby("kind").agg(
    max_uncapped_a=("peak_a_uncapped", "max"), max_capped_a=("peak_a", "max"),
    frac_capped=("capped", "mean")).reset_index(),
    "Reference feasibility summary (spd = 1.5 m/s)",
    note=f"budget a_lat_max = {X.A_LAT_MAX:.3f}, cap alpha*a_lat = "
         f"{X.P.alpha_feas*X.A_LAT_MAX:.4f} m/s^2",
    csv=("common", "path_feasibility.csv"))
PF.to_csv(X.apath("common", "path_feasibility.csv"), index=False)

fig, ax = plt.subplots(figsize=(6.2, 4.0))
mk = dict(zip(X.PATHS_ALL, ["o", "s", "^", "v", "D", "P"]))
for i, kind in enumerate(X.PATHS_ALL):
    d = PF[PF.kind == kind]
    if d.peak_a_uncapped.max() <= 0:
        continue
    ax.plot(d.R, d.peak_a_uncapped, "--", color=f"C{i}", marker=mk[kind],
            markevery=4, ms=4, alpha=0.7)
    ax.plot(d.R, d.peak_a, "-", color=f"C{i}", marker=mk[kind], markevery=4,
            ms=4, label=kind)
ax.axhline(X.A_LAT_MAX, color="k", lw=1.2)
ax.text(1.05, X.A_LAT_MAX * 1.02, f"$a_{{lat}}^{{max}}$ = {X.A_LAT_MAX:.2f}", fontsize=7)
ax.axhline(X.P.alpha_feas * X.A_LAT_MAX, color="k", ls=":", lw=1.2)
ax.text(1.05, X.P.alpha_feas * X.A_LAT_MAX * 1.03,
        rf"$\alpha a_{{lat}}^{{max}}$ = {X.P.alpha_feas*X.A_LAT_MAX:.2f}", fontsize=7)
ax.set_yscale("log"); ax.set_xlabel("path radius $R$ [m]")
ax.set_ylabel(r"peak $\|a_{ref}\|$ [m/s$^2$]  (log)")
ax.set_title("F1  reference feasibility: dashed = uncapped, solid = capped (4.6)")
ax.legend(fontsize=7, ncol=2)
fig.savefig(f"{FIG}/F1_feasibility.png", bbox_inches="tight")
PF.to_csv(f"{FIG}/F1_feasibility.csv", index=False)

worst = PF.loc[PF.peak_a_uncapped.idxmax()]
print(f"\n  ANALYSIS. Uncapped, {worst['kind']} at R = {worst['R']:.2f} demands "
      f"{worst['peak_a_uncapped']:.2f} m/s^2, which is "
      f"{worst['peak_a_uncapped']/X.A_LAT_MAX:.2f}x the whole envelope and "
      f"{worst['peak_a_uncapped']/(X.P.alpha_feas*X.A_LAT_MAX):.2f}x the cap.")
for kind in X.PATHS_ALL:
    d = PF[(PF.kind == kind) & PF.capped]
    if len(d):
        f = (d.peak_a_uncapped / (X.P.alpha_feas * X.A_LAT_MAX)).max()
        print(f"    {kind:7s}: capped on {100*PF[PF.kind==kind].capped.mean():5.1f}% "
              f"of radii, worst overshoot {f:.2f}x")

# %% [markdown]
# ## Step 3 — the classical anchor: LQR
#
# Two rows per path, with and without the reference feed-forward (4.8). The gap
# between the columns is the feed-forward term alone.
#
# PID is **not** in this notebook: it shares no design object with anything else
# here, so its column would measure how well someone tuned nine gains.

# %%
S.section(3, "LQR baselines", "the classical anchor, with and without feed-forward",
          produces="lltc/nb1_classical.csv, T1, F2")

PATHS_EVAL = ("hover", "circle", "fig8", "square")
rowsL = []
for kind in PATHS_EVAL:
    for name, ctrl in (("LQR", X.make_lqr_ctrl()),
                       ("LQR (no feed-fwd)", X.make_lqr_ctrl(feedforward=False))):
        env = S.ev_env(kind, spec=S.nominal_spec(speed=(1.0, 1.5)), seed=101)
        st = X.stats(X.rollout_eval(env, ctrl, S.CFG["T_eval"], warmup=50))
        rowsL.append(dict(path=kind, ctrl=name, **st))
T1 = pd.DataFrame(rowsL)
S.table(T1[["path", "ctrl", "rmse", "rmse_iqr", "maxerr", "tilt", "effort",
            "smooth", "sat", "crash"]],
        "T1  LQR baselines by path",
        note="maxerr ~ 0.49 is the seeded initial offset e_p(0) ~ U([-0.5,0.5]^3), "
             "not a tracking property",
        csv=("lltc", "nb1_classical.csv"))

fig, ax = plt.subplots(figsize=(5.6, 3.4))
w = 0.38
xs = np.arange(len(PATHS_EVAL))
for j, name in enumerate(("LQR", "LQR (no feed-fwd)")):
    d = T1[T1.ctrl == name].set_index("path").loc[list(PATHS_EVAL)]
    ax.bar(xs + (j - 0.5) * w, d.rmse, w, label=name,
           hatch=["", "//"][j], edgecolor="k", lw=0.5)
ax.set_xticks(xs); ax.set_xticklabels(PATHS_EVAL)
ax.set_ylabel("position RMSE [m]"); ax.legend(fontsize=7)
ax.set_title("F2  LQR with and without the reference feed-forward (4.8)")
fig.savefig(f"{FIG}/F2_lqr_feedforward.png", bbox_inches="tight")
T1.to_csv(f"{FIG}/F2_lqr_feedforward.csv", index=False)

piv = T1.pivot(index="path", columns="ctrl", values="rmse")
print("\n  ANALYSIS. The gap is the feed-forward term alone (same gain, same model):")
for k in PATHS_EVAL:
    a, b = piv.loc[k, "LQR"], piv.loc[k, "LQR (no feed-fwd)"]
    print(f"    {k:7s}: {a:.4f} -> {b:.4f} m   ({100*(b-a)/max(a,1e-9):+.1f}%)")
print(f"    maxerr across all rows: {T1.maxerr.min():.3f} .. {T1.maxerr.max():.3f} m")

# %% [markdown]
# ## Step 4 — horizon sweep, preview against a genuinely frozen reference
#
# **Freezing the reference is a bug, not a simplification.** At 1.5 m/s over a
# 20-step horizon the target moves 0.6 m, so a frozen-reference solver plans to
# come to rest on the current waypoint and error *grows* with $N$.
#
# The frozen controller is a closure with **no `bind_env`**, so `rollout_eval`
# cannot hand it the preview. The assertion below is T-6.

# %%
S.section(4, "horizon sweep", "what an online solve buys, and what it costs",
          produces="common/nb1_horizon.csv, F3, F4")

NS = (1, 2, 3, 5, 10, 20)
TRAJ = ("circle", "fig8", "square")
HZ = {"N": list(NS)}
LAT = {"ms_median": [], "ms_p95": []}
for kind in TRAJ:
    HZ[kind], HZ[kind + "_frozen"] = [], []
for N in NS:
    for kind in TRAJ:
        env = S.ev_env(kind, spec=S.nominal_spec(speed=(1.0, 1.5)), seed=202)
        c = X.make_nmpc_ctrl(N=N, n_iter=S.CFG["ilqr"])
        HZ[kind].append(X.stats(X.rollout_eval(env, c, S.CFG["T_eval"] // 2,
                                               warmup=50))["rmse"])
        envf = S.ev_env(kind, spec=S.nominal_spec(speed=(1.0, 1.5)), seed=202)
        cf = X.make_frozen_nmpc_ctrl(N=N, n_iter=S.CFG["ilqr"])
        assert not hasattr(cf, "bind_env"), "the frozen controller must not bind"
        HZ[kind + "_frozen"].append(
            X.stats(X.rollout_eval(envf, cf, S.CFG["T_eval"] // 2, warmup=50))["rmse"])
    env1 = S.ev_env("circle", n=1, spec=S.nominal_spec(speed=(1.2, 1.2)), seed=7)
    lat = X.solve_latency_ms(X.make_nmpc_ctrl(N=N, n_iter=S.CFG["ilqr"]), env1, T=30)
    LAT["ms_median"].append(lat["median"]); LAT["ms_p95"].append(lat["p95"])
HZdf = pd.DataFrame({**HZ, **LAT})

same = [k for k in TRAJ if np.allclose(HZdf[k], HZdf[k + "_frozen"], rtol=1e-6)]
assert not same, f"the frozen control is still receiving the preview: {same}"
S.table(HZdf, "Horizon sweep: preview vs genuinely frozen reference",
        note="single-vehicle latency (solve_latency_ms), NOT ms_per_step/batch",
        csv=("common", "nb1_horizon.csv"))

fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.6))
for i, kind in enumerate(TRAJ):
    axs[0].plot(HZdf.N, HZdf[kind], "-o", color=f"C{i}", ms=4, label=f"{kind} (preview)")
    axs[0].plot(HZdf.N, HZdf[kind + "_frozen"], "--s", color=f"C{i}", ms=4, alpha=0.8,
                label=f"{kind} (frozen)")
axs[0].set_xlabel("horizon $N$"); axs[0].set_ylabel("position RMSE [m]")
axs[0].set_title("(a) accuracy vs horizon"); axs[0].legend(fontsize=6.5, ncol=2)
axs[1].semilogy(HZdf.N, HZdf.ms_median, "-o", ms=4, label="median")
axs[1].semilogy(HZdf.N, HZdf.ms_p95, "--s", ms=4, label="p95")
axs[1].axhline(20.0, color="C3", lw=1.4)
axs[1].text(1.2, 22, "20 ms period", color="C3", fontsize=7)
axs[1].set_xlabel("horizon $N$"); axs[1].set_ylabel("single-vehicle solve [ms] (log)")
axs[1].set_title("(b) latency vs horizon"); axs[1].legend(fontsize=7)
fig.suptitle("F3  horizon sweep", fontsize=10)
fig.savefig(f"{FIG}/F3_horizon.png", bbox_inches="tight")
HZdf.to_csv(f"{FIG}/F3_horizon.csv", index=False)

cross = HZdf[HZdf.ms_p95 > 20.0]
print("\n  ANALYSIS.")
for kind in TRAJ:
    p, f = HZdf[kind].values, HZdf[kind + "_frozen"].values
    print(f"    {kind:7s}: preview {p[0]:.4f} (N=1) -> {p[-1]:.4f} (N=20), "
          f"{'falls' if p[-1] < p[0] else 'RISES'}; "
          f"frozen {f[0]:.4f} -> {f[-1]:.4f}, "
          f"{'grows' if f[-1] > f[0] else 'does NOT grow'}")
print(f"    p95 latency crosses 20 ms at "
      f"{'N = ' + str(int(cross.N.iloc[0])) if len(cross) else 'no tested horizon'}"
      f" (max tested p95 {HZdf.ms_p95.max():.2f} ms).")
_rising = [k for k in TRAJ if HZdf[k].values[-1] > HZdf[k].values[0] * 1.05]
if _rising:
    print(f"    NOTE: the PREVIEW column rises with N on {_rising}.  That is not "
          f"the frozen-reference mechanism.  On a path at the feasibility cap "
          f"the collective is near its box (sat = "
          f"{T1[T1.path=='square'].sat.max():.2f} for LQR on the superellipse), "
          f"so a longer plan is clipped inside the rollout and the solve is "
          f"harder; with ilqr = {S.CFG['ilqr']} iterations it may also be "
          f"under-converged.  Re-read this column at X500_SCALE=full "
          f"(ilqr = {S.SCALES['full']['ilqr']}) before drawing a conclusion "
          f"about horizon length from it.")

# F4: corner detail on the superellipse
env = S.ev_env("square", n=1, spec=S.nominal_spec(speed=(1.4, 1.4)), seed=303,
               ep_len=100000)
env.fixed = dict(R=0.9); env.reset()
d1 = V.fly(env, X.make_nmpc_ctrl(N=1, n_iter=S.CFG["ilqr"]),
           V.clip_length(env, 1.0)[0])
env2 = S.ev_env("square", n=1, spec=S.nominal_spec(speed=(1.4, 1.4)), seed=303,
                ep_len=100000)
env2.fixed = dict(R=0.9); env2.reset()
d10 = V.fly(env2, X.make_nmpc_ctrl(N=10, n_iter=S.CFG["ilqr"]),
            V.clip_length(env2, 1.0)[0])
fig, ax = plt.subplots(figsize=(4.8, 4.6))
ax.plot(d1["RFULL"][:, 0], d1["RFULL"][:, 1], "--", color="0.4", label="reference")
ax.plot(d1["p"][:, 0], d1["p"][:, 1], "-o", ms=2.5, markevery=9, label="N=1")
ax.plot(d10["p"][:, 0], d10["p"][:, 1], "-s", ms=2.5, markevery=9, label="N=10")
ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title(f"F4  superellipse corner: N=1 (RMSE {d1['rmse']:.3f}) vs "
             f"N=10 ({d10['rmse']:.3f})", fontsize=8)
ax.legend(fontsize=7)
fig.savefig(f"{FIG}/F4_corner.png", bbox_inches="tight")
pd.DataFrame(dict(x1=d1["p"][:, 0], y1=d1["p"][:, 1],
                  x10=d10["p"][:len(d1["p"]), 0],
                  y10=d10["p"][:len(d1["p"]), 1])).to_csv(f"{FIG}/F4_corner.csv",
                                                          index=False)

# %% [markdown]
# ## Step 5 — noise sensitivity  {#sec:noise}

# %%
S.section(5, "noise sensitivity", "how the baselines degrade with measurement noise",
          produces="common/nb1_noise.csv")

rowsN = []
for noise in ("off", "low", "high"):
    for name, mk_ in (("LQR", lambda: X.make_lqr_ctrl()),
                      ("NMPC N=1", lambda: X.make_nmpc_ctrl(N=1, n_iter=S.CFG["ilqr"])),
                      ("NMPC N=10", lambda: X.make_nmpc_ctrl(N=10, n_iter=S.CFG["ilqr"]))):
        env = S.ev_env("circle", spec=S.nominal_spec(speed=(1.0, 1.5)), seed=404,
                       noise=noise)
        st = X.stats(X.rollout_eval(env, mk_(), S.CFG["T_eval"] // 2, warmup=50))
        rowsN.append(dict(noise=noise, ctrl=name, **st))
NZ = pd.DataFrame(rowsN)
S.table(NZ[["noise", "ctrl", "rmse", "maxerr", "smooth", "sat", "crash"]],
        "Noise sensitivity (sec:noise)",
        note=f"levels: {X.NOISE_LEVELS}", csv=("common", "nb1_noise.csv"))
fig, ax = plt.subplots(figsize=(5.4, 3.3))
for i, c in enumerate(NZ.ctrl.unique()):
    d = NZ[NZ.ctrl == c]
    ax.plot(d.noise, d.rmse, "-o", ms=4, color=f"C{i}", label=c)
ax.set_ylabel("position RMSE [m]"); ax.set_xlabel("measurement noise")
ax.legend(fontsize=7); ax.set_title("noise sensitivity of the baselines")
fig.savefig(f"{FIG}/F3b_noise.png", bbox_inches="tight")
NZ.to_csv(f"{FIG}/F3b_noise.csv", index=False)
print("\n  ANALYSIS. degradation factor high/off:")
for c in NZ.ctrl.unique():
    d = NZ[NZ.ctrl == c].set_index("noise")
    print(f"    {c:10s}: {d.loc['high','rmse']/max(d.loc['off','rmse'],1e-9):.2f}x")

# %% [markdown]
# ## Step 6 — the cost-weight sweep
#
# This is the measurement the whole study exists to remove. The ratio of the
# best admissible weighting to the worst is the fraction of "controller
# performance" that is really weight tuning.

# %%
S.section(6, "Q/R sweep", "how much of the result is weight tuning",
          produces="common/nb1_qr_sweep.csv, F5, T2")

QP = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
RW = np.array([0.05, 0.2, 1.0, 5.0, 20.0])
rowsQ = []
for qp in QP:
    for rw in RW:
        Q = np.diag([qp, qp, qp, 0.4 * qp, 0.4 * qp, 0.4 * qp, 1.0, 1.0, 0.5])
        R = np.diag([0.5, rw, rw, rw])
        try:
            _, Pi, _, _ = X.dlqr(Q, R)
            env = S.ev_env(("circle", "fig8"), n=min(S.CFG["n_eval"], 32),
                           spec=S.nominal_spec(speed=(1.0, 1.5)), seed=505)
            st = X.stats(X.rollout_eval(
                env, X.make_nmpc_ctrl(N=1, n_iter=S.CFG["ilqr"], Q=Q, R=R,
                                      Pterm=jnp.asarray(Pi)),
                S.CFG["T_eval"] // 3, warmup=40))
            rowsQ.append(dict(Q_pos=qp, R_rate=rw, admissible=st["crash"] < 0.02, **st))
        except Exception as ex:
            rowsQ.append(dict(Q_pos=qp, R_rate=rw, admissible=False, rmse=np.nan,
                              maxerr=np.nan, sat=np.nan, crash=np.nan))
QR = pd.DataFrame(rowsQ)
S.table(QR[["Q_pos", "R_rate", "rmse", "maxerr", "sat", "crash", "admissible"]],
        "Q/R sweep, NMPC N=1, nominal", csv=("common", "nb1_qr_sweep.csv"))

adm = QR[QR.admissible & QR.rmse.notna()]
best, worst = adm.rmse.min(), adm.rmse.max()
T2 = pd.DataFrame([
    dict(case="best", Q_pos=adm.loc[adm.rmse.idxmin(), "Q_pos"],
         R_rate=adm.loc[adm.rmse.idxmin(), "R_rate"], rmse=best),
    dict(case="median", Q_pos=np.nan, R_rate=np.nan, rmse=adm.rmse.median()),
    dict(case="worst", Q_pos=adm.loc[adm.rmse.idxmax(), "Q_pos"],
         R_rate=adm.loc[adm.rmse.idxmax(), "R_rate"], rmse=worst),
    dict(case="best-to-worst ratio", Q_pos=np.nan, R_rate=np.nan,
         rmse=worst / max(best, 1e-12)),
])
S.table(T2, "T2  Q/R spread over the admissible grid",
        note="the ratio is the fraction of 'controller performance' that is "
             "really weight tuning -- the number the rest of the study removes",
        csv=("common", "nb1_qr_spread.csv"))

fig, ax = plt.subplots(figsize=(5.2, 4.0))
Z = QR.pivot(index="Q_pos", columns="R_rate", values="rmse").values
im = ax.imshow(np.log10(Z), origin="lower", aspect="auto", cmap="viridis")
ax.set_xticks(range(len(RW))); ax.set_xticklabels([f"{v:g}" for v in RW])
ax.set_yticks(range(len(QP))); ax.set_yticklabels([f"{v:g}" for v in QP])
ax.set_xlabel("R (rate channels)"); ax.set_ylabel("Q (position)")
op_q = float(X.Q_HAND[0, 0]); op_r = float(X.R_HAND[1, 1])
ax.plot(list(RW).index(op_r), list(QP).index(op_q), "x", color="w", ms=14, mew=3)
ax.text(list(RW).index(op_r) + 0.12, list(QP).index(op_q) + 0.12,
        "operating point", color="w", fontsize=7)
plt.colorbar(im, ax=ax, label=r"$\log_{10}$ RMSE [m]")
ax.set_title("F5  cost-weight sweep, NMPC N=1")
fig.savefig(f"{FIG}/F5_qr_heatmap.png", bbox_inches="tight")
QR.to_csv(f"{FIG}/F5_qr_heatmap.csv", index=False)

print(f"\n  ANALYSIS. Over the admissible grid, closed-loop RMSE runs "
      f"{best:.4f} m to {worst:.4f} m -- a best-to-worst ratio of "
      f"{worst/max(best,1e-12):.1f}x.")
print(f"    Nothing in the theory determines Q and R.  A human picks them, and "
      f"that choice moves the result by {worst/max(best,1e-12):.1f}x.")
print(f"    The hand-chosen operating point (Q_pos={op_q:g}, R={op_r:g}) sits at "
      f"RMSE {adm[(adm.Q_pos==op_q)&(adm.R_rate==op_r)].rmse.iloc[0]:.4f} m.")

# %% [markdown]
# ## Step 7 — ledger row

# %%
S.section(7, "ledger", "append LQR", produces="common/ledger.csv")
env1 = S.ev_env("circle", n=1, spec=S.nominal_spec(speed=(1.2, 1.2)), seed=7)
lat = X.solve_latency_ms(X.make_lqr_ctrl(), env1, T=40)
lqr_row = T1[(T1.ctrl == "LQR")].rmse.mean()
LED = pd.DataFrame([dict(ctrl="LQR", rmse_S1=float(lqr_row),
                         ms_median=lat["median"], ms_p95=lat["p95"],
                         tuned=X.NE + X.NU, scale=S.SCALE)])
S.table(LED, "Ledger (first row)",
        note="tuned = N_e + N_u = 13 hand-chosen stage weights",
        csv=("common", "ledger.csv"))

# %% [markdown]
# ## What this notebook establishes
#
# Every statement below is printed by a cell above, from this run.
#
# 1. **The constants are right.** All 17 self-test rows match §2 to 1e-6, and
#    the residual and wrench are identically zero on the undisturbed plant.
# 2. **The uncapped references were not flyable.** The superellipse at small
#    radius demands several times the whole lateral envelope; without (4.6)
#    every controller fails identically and the benchmark measures the
#    reference.
# 3. **The feed-forward term is not optional.** Removing (4.8) alone moves the
#    LQR row, with the same gain and the same model.
# 4. **A preview is not a luxury.** With the reference frozen, error behaves
#    qualitatively differently from the preview case — the assertion in step 4
#    is what proves the two columns are actually different experiments.
# 5. **And the headline: weight tuning moves the answer by the ratio printed in
#    T2.** That ratio is what Notebooks 2 and 3 are trying to remove.
#
# `maxerr ≈ 0.49` wherever it appears is the seeded initial offset, not a
# tracking property.

# %%
S.checkpoint(feasibility=("common", "path_feasibility.csv"),
             classical=("lltc", "nb1_classical.csv"),
             horizon=("common", "nb1_horizon.csv"),
             qr_sweep=("common", "nb1_qr_sweep.csv"),
             ledger=("common", "ledger.csv"))
print("\nNotebook 1 complete.")
