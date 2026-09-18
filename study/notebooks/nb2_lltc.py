# %% [markdown]
# # Notebook 2 — LLTC, the learned terminal cost
#
# **Establishes.** Whether a learned quadratic terminal cost gives a one-step
# horizon the behaviour of a long one, and whether the object generalises.
#
# **Consumes.** `common/nb1_horizon.csv` (for the NMPC reference columns).
#
# **Produces.** `lltc/model.pkl`, `lltc/nb2_fit.csv`,
# `lltc/nb2_horizon_equivalence.csv`, `lltc/nb2_ood.csv`,
# `lltc/sensitivity/qr_scalar.csv`, figures F6–F8, tables T3–T4.
#
# $\mathbf P_\theta(\mathbf e_0)=\mathbf L_\theta\mathbf L_\theta^\top+\varepsilon\mathbf I$
# is PSD by construction, so (5.2) stays well posed for any $\theta$.

# %%
import _nbinit  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax

import x500_core_jax as X
import study_prelude as S
import study_moderate as M
import viz as V

S.header("Notebook 2 - LLTC, the learned terminal cost")
X.style()
FIG = V.figures_dir()
KEY = jax.random.PRNGKey(2)

# %% [markdown]
# ## Data generation
#
# Sample candidate errors around the reference, run a long-horizon solve, and
# keep those whose realised cost-to-go $V_1 = J - \ell_0$ lies inside the
# terminal set.
#
# **The terminal-set reach and the acceptance rate are reported side by side.**
# If `disp` is several times the reach, most candidates lie where the local
# quadratic model does not hold, and the two must not be described as matched.

# %%
S.section(1, "candidate generation", "sample errors and price their cost-to-go",
          produces="lltc/nb2_fit.csv")

N_LONG = 10
DISP = S.CFG["disp"]
N_CAND = S.CFG["n_cand"]
env = S.ev_env("circle", n=min(N_CAND, 256), spec=S.nominal_spec(speed=(1.0, 1.5)),
               seed=11)
o, e0, xr = env.obs()
B = env.n
KEY, k1, k1b = jax.random.split(KEY, 3)
scale = jnp.asarray([DISP, DISP, DISP, DISP, DISP, DISP,
                     0.3 * DISP, 0.3 * DISP, 0.3 * DISP])
# G7: sample over DECADES of error magnitude, not one shell at `disp`.
# P_theta is an MLP of e, so it is only constrained where candidates were drawn.
# Drawn at |e| ~ 0.8 m alone, the closed loop -- which lives at |e| ~ 0.05 m --
# queries the network a factor ~15 inside its support, i.e. pure extrapolation,
# which is a candidate explanation for LLTC's 26 % saturation next to NMPC
# N=1's 0.07 %.
decade = 10.0 ** jax.random.uniform(k1b, (B, 1), minval=-1.5, maxval=0.0)
cand = jax.random.normal(k1, (B, X.NE)) * scale * decade
xr_seq, uref_seq = env.ref_traj(N_LONG), env.ref_useq(N_LONG)
Sm, cm = X.quad_cost_blocks(B, N_LONG)
Pt = jnp.broadcast_to(X.PTt, (B, X.NE, X.NE))
du = X.ilqr_solve(cand, xr_seq, Sm, cm, Pt, None, None, S.CFG["ilqr"], 0, uref_seq)
e_seq = X.rollout_err(cand, du, xr_seq, uref_seq)
J = X.traj_cost(e_seq, du, Sm, cm, Pt)
tau0 = jnp.concatenate([e_seq[:, 0], du[:, 0]], -1)
l0 = 0.5 * jnp.einsum("bi,bij,bj->b", tau0, Sm[:, 0], tau0)
V1 = np.asarray(J - l0)

# terminal-set reach: the error radius at which the terminal quadratic still
# describes the realised cost-to-go
REACH = float(np.sqrt(2.0 * np.median(V1) / max(float(X.P_RIC[0, 0]), 1e-9)))
# G7b: the gate must NOT depend on the candidate draw.  G7 filtered on
# ||e1|| <= 2*REACH while REACH is itself sqrt(2*median(V1)/P_00) -- computed
# from the very candidates being filtered.  Adding the decade sampling dropped
# the median, REACH fell 0.556 -> 0.086 m, the gate then kept only the
# small-error candidates, and that fit was deployed against a seeded 0.5 m
# offset: LLTC went 0.41 -> 1.22 m, horizon equivalence 7.5x -> 23.7x, and the
# clip diverged to 35.6 m.  It INVERTED the original defect (fit large, deploy
# small) instead of removing it.  A realised cost-to-go that is finite and
# positive is a real criterion and is all that is needed; REACH stays a
# reported diagnostic, never a filter.
keep = np.isfinite(V1) & (V1 > 0.0)
accept = float(keep.mean())
# G7: fit the object the CONTROLLER evaluates.  V1 is the cost-to-go from
# e_1, and `make_lltc_ctrl` scores 0.5 e_1' P(e_0) e_1 as the terminal cost of
# the N=1 problem.  Regressing 0.5 e_0' P(e_0) e_0 onto V1 -- the old code --
# fits a different quadratic form from the one that is later used, so the near-
# perfect R^2 was never a statement about the deployed object.
E_fit = np.asarray(cand)[keep]          # the argument of P_theta
E1_fit = np.asarray(e_seq[:, 1])[keep]  # what the quadratic is contracted with
V_fit = V1[keep]
print(f"  candidates {B}, accepted {keep.sum()} ({100*accept:.1f}%)")
print(f"  displacement scale CFG['disp'] = {DISP:.3f} m")
print(f"  terminal-set reach              = {REACH:.3f} m")
print(f"  ratio disp/reach                = {DISP/max(REACH,1e-9):.2f}")
if DISP > 2.0 * REACH:
    print("  !! disp is more than 2x the reach: most candidates lie where the "
          "local quadratic model does not hold.  These are NOT matched.")

# %% [markdown]
# ## Training and the quality gate
#
# $R^2 \le 0$ means the regression is worse than predicting the mean of $V_1$.
# That is a **negative result about the fit**, and the horizon-equivalence claim
# downstream is then unsupported. It is reported as such, not tuned away.

# %%
S.section(2, "fit", "regress the terminal quadratic onto the realised cost-to-go",
          produces="lltc/model.pkl, T3, F6")

KEY, k2 = jax.random.split(KEY)
params = M.lltc_init(k2, hid=max(S.CFG["hid"] // 4, 32))
EPS = 1e-3
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
st = opt.init(params)
Ej, Vj = jnp.asarray(E_fit), jnp.asarray(V_fit)
E1j = jnp.asarray(E1_fit)

def loss_fn(p, ee, e1, vv):
    P = M.lltc_matrix(p, ee, EPS)                       # (8.1), evaluated at e_0
    pred = 0.5 * jnp.einsum("bi,bij,bj->b", e1, P, e1)  # contracted with e_1
    return jnp.mean((pred - vv) ** 2)                   # (8.2)

@jax.jit
def upd(p, st, ee, e1, vv):
    l, g = jax.value_and_grad(loss_fn)(p, ee, e1, vv)
    u, st = opt.update(g, st, p)
    return optax.apply_updates(p, u), st, l

n_ep = 400 if S.SCALE != "smoke" else 120
hist = []
for ep in range(n_ep):
    params, st, l = upd(params, st, Ej, E1j, Vj)
    hist.append(float(l))

P_of = M.lltc_matrix(params, Ej, EPS)
pred = np.asarray(0.5 * jnp.einsum("bi,bij,bj->b", E1j, P_of, E1j))
ss_res = float(((V_fit - pred) ** 2).sum())
ss_tot = float(((V_fit - V_fit.mean()) ** 2).sum())
R2 = 1.0 - ss_res / max(ss_tot, 1e-30)
NRMSE = float(np.sqrt(ss_res / len(V_fit)) / max(V_fit.std(), 1e-12))
FIT = dict(R2=R2, NRMSE=NRMSE, accept=accept, n=int(keep.sum()), reach=REACH,
           disp=DISP, final_loss=hist[-1])

LLTC_OK = FIT["R2"] > 0.0
if not LLTC_OK:
    print(f"!! R2 = {FIT['R2']:+.3f} <= 0: the fit is worse than predicting "
          f"the mean of V1.  Acceptance {100*accept:.1f}%, n={FIT['n']}. "
          f"Levers: CFG['n_cand']={N_CAND} (acceptance binds) and "
          f"CFG['disp']={DISP} against terminal-set reach {REACH:.3f} m.")

T3 = pd.DataFrame([dict(R2=R2, NRMSE=NRMSE, acceptance=accept, n=FIT["n"],
                        reach_m=REACH, disp_m=DISP,
                        disp_over_reach=DISP / max(REACH, 1e-9), gate_passed=LLTC_OK)])
S.table(T3, "T3  LLTC fit diagnostics",
        note="R2 <= 0 is a negative result about the fit, not a tuning target",
        csv=("lltc", "nb2_fit.csv"))
X.save_ckpt(dict(state=params, arch=dict(hid=max(S.CFG["hid"] // 4, 32)),
                 ltc=dict(eps=EPS), fit=FIT), "lltc", "model.pkl")

fig, ax = plt.subplots(figsize=(4.4, 4.2))
ax.scatter(V_fit, pred, s=6, alpha=0.4)
lim = [min(V_fit.min(), pred.min()), max(V_fit.max(), pred.max())]
ax.plot(lim, lim, "--", color="0.3", label="identity")
ax.set_xlabel("realised $V_1$"); ax.set_ylabel(r"predicted $\frac{1}{2} e^T P_\theta e$")
ax.set_title(f"F6  LLTC fit, $R^2$ = {R2:+.3f}")
ax.legend(fontsize=7)
fig.savefig(f"{FIG}/F6_lltc_fit.png", bbox_inches="tight")
pd.DataFrame(dict(realised=V_fit, predicted=pred)).to_csv(
    f"{FIG}/F6_lltc_fit.csv", index=False)

# %% [markdown]
# ## Evaluation 1 — horizon equivalence
#
# LLTC at $N=1$ against NMPC at $N\in\{1,3,10\}$. **A ratio below 1.0 supports
# the claim** that a learned terminal cost buys a one-step horizon the behaviour
# of a long one.

# %%
S.section(3, "horizon equivalence", "does N=1 + learned terminal cost act like N=10?",
          produces="lltc/nb2_horizon_equivalence.csv, T4")

model = X.load_ckpt("lltc", "model.pkl")
rows = []
for kind in ("circle", "fig8", "square"):
    r = {}
    for name, ctrl in (("LLTC N=1", M.make_lltc_ctrl(X, model, N=1, n_iter=S.CFG["ilqr"])),
                       ("NMPC N=1", X.make_nmpc_ctrl(N=1, n_iter=S.CFG["ilqr"])),
                       ("NMPC N=3", X.make_nmpc_ctrl(N=3, n_iter=S.CFG["ilqr"])),
                       ("NMPC N=10", X.make_nmpc_ctrl(N=10, n_iter=S.CFG["ilqr"]))):
        env = S.ev_env(kind, spec=S.nominal_spec(speed=(1.0, 1.5)), seed=606)
        r[name] = X.stats(X.rollout_eval(env, ctrl, S.CFG["T_eval"] // 2,
                                         warmup=50))["rmse"]
    r["ratio LLTC/NMPC N=1"] = r["LLTC N=1"] / max(r["NMPC N=1"], 1e-12)
    r["ratio LLTC/NMPC N=10"] = r["LLTC N=1"] / max(r["NMPC N=10"], 1e-12)
    rows.append(dict(path=kind, **r))
T4 = pd.DataFrame(rows)
S.table(T4, "T4  horizon equivalence (position RMSE [m])",
        note="ratio below 1.0 supports the claim; above 1.0 refutes it",
        csv=("lltc", "nb2_horizon_equivalence.csv"))
VERDICT = ("SUPPORTED" if (T4["ratio LLTC/NMPC N=1"] < 1.0).all() else
           "NOT SUPPORTED on every path")
print(f"\n  VERDICT (computed from the table above): horizon equivalence is "
      f"{VERDICT}.")
for _, r in T4.iterrows():
    print(f"    {r['path']:7s}: LLTC/NMPC(N=1) = {r['ratio LLTC/NMPC N=1']:.3f}, "
          f"LLTC/NMPC(N=10) = {r['ratio LLTC/NMPC N=10']:.3f}")
if not LLTC_OK:
    print("    ...but the fit gate failed, so this claim is UNSUPPORTED "
          "regardless of the ratios: a terminal cost fitted worse than a "
          "constant is not evidence about horizons.")

# %% [markdown]
# ## Evaluation 2 — locality and OOD
#
# $\mathbf P_\theta$ is fitted where the data is. Out of distribution it carries
# no positive-definiteness guarantee beyond the $\varepsilon I$ floor and no
# dissipativity guarantee at all.

# %%
S.section(4, "locality / OOD", "how far does the fitted object travel",
          produces="lltc/nb2_ood.csv, F7b")

S2 = M.moderate(S.disturbed_spec(), wind=(0.0, 2.0))
S3 = M.moderate(S.ood_spec(), wind=(2.5, 4.0))
M.check_moderate(S.disturbed_spec(), S2, 2.0)
M.check_moderate(S.ood_spec(), S3, 4.0)
print(M.disjointness(S2, S3).to_string(index=False))
print("  -> disjoint on drag, lag and wind ONLY; mass, thrust, K_w and inertia "
      "overlap (C-4).  Do not claim full disjointness.")

rowsO = []
for suite, spec in (("S1 nominal", S.nominal_spec(speed=(1.0, 1.5))),
                    ("S2 moderated in-dist", S2), ("S3 moderated OOD", S3)):
    for name, ctrl in (("LLTC N=1", M.make_lltc_ctrl(X, model, N=1, n_iter=S.CFG["ilqr"])),
                       ("NMPC N=1", X.make_nmpc_ctrl(N=1, n_iter=S.CFG["ilqr"]))):
        env = S.ev_env(("circle", "fig8"), spec=spec, seed=707)
        rowsO.append(dict(suite=suite, ctrl=name,
                          **X.stats(X.rollout_eval(env, ctrl, S.CFG["T_eval"] // 2,
                                                   warmup=50))))
OOD = pd.DataFrame(rowsO)
S.table(OOD[["suite", "ctrl", "rmse", "maxerr", "sat", "crash"]],
        "LLTC in and out of distribution",
        note="report saturation beside RMSE: if the path stops mattering you "
             "are measuring the disturbance",
        csv=("lltc", "nb2_ood.csv"))

# %% [markdown]
# ## Evaluation 3 — $Q_p$ sensitivity

# %%
S.section(5, "Qp sensitivity", "reach and acceptance as the terminal weighting moves",
          produces="lltc/sensitivity/qr_scalar.csv, F7")

rowsQ = []
for qs in (0.25, 0.5, 1.0, 2.0, 4.0):
    Q = X.Q_HAND * qs
    _, Pi, _, _ = X.dlqr(Q, X.R_HAND)
    reach_q = float(np.sqrt(2.0 * np.median(V1) / max(float(Pi[0, 0]), 1e-9)))
    Sm_q, cm_q = X.quad_cost_blocks(B, N_LONG, Q=Q)
    duq = X.ilqr_solve(cand, xr_seq, Sm_q, cm_q,
                       jnp.broadcast_to(jnp.asarray(Pi), (B, X.NE, X.NE)),
                       None, None, S.CFG["ilqr"], 0, uref_seq)
    esq = X.rollout_err(cand, duq, xr_seq, uref_seq)
    Jq = np.asarray(X.traj_cost(esq, duq, Sm_q, cm_q,
                                jnp.broadcast_to(jnp.asarray(Pi), (B, X.NE, X.NE))))
    # G7b: same gate as the fit -- finite and positive, never filtered on a
    # radius derived from the candidates themselves.
    acc_q = float((np.isfinite(Jq) & (Jq > 0.0)).mean())
    rowsQ.append(dict(Q_scale=qs, P_00=float(Pi[0, 0]), reach_m=reach_q,
                      acceptance=acc_q))
QS = pd.DataFrame(rowsQ)
S.table(QS, "Terminal-weight sensitivity",
        csv=("lltc", "sensitivity", "qr_scalar.csv"))

fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4))
axs[0].plot(QS.Q_scale, QS.reach_m, "-o", ms=4, label="terminal-set reach [m]")
ax2 = axs[0].twinx()
ax2.plot(QS.Q_scale, QS.acceptance, "--s", color="C1", ms=4, label="acceptance")
axs[0].set_xlabel("$Q_p$ scale"); axs[0].set_ylabel("reach [m]")
ax2.set_ylabel("acceptance [-]")
axs[0].axhline(DISP, color="0.4", ls=":", lw=1)
axs[0].text(QS.Q_scale.min(), DISP * 1.02, f"disp = {DISP:g}", fontsize=7)
axs[0].set_title("(a) acceptance and reach vs $Q_p$")
piv = OOD.pivot(index="suite", columns="ctrl", values="rmse")
xs = np.arange(len(piv))
for j, c in enumerate(piv.columns):
    axs[1].bar(xs + (j - 0.5) * 0.38, piv[c], 0.38, label=c,
               hatch=["", "//"][j % 2], edgecolor="k", lw=0.5)
axs[1].set_xticks(xs); axs[1].set_xticklabels(piv.index, fontsize=7)
axs[1].set_ylabel("RMSE [m]"); axs[1].legend(fontsize=7)
axs[1].set_title("(b) in-distribution vs OOD")
fig.suptitle("F7  LLTC locality", fontsize=10)
fig.savefig(f"{FIG}/F7_lltc_locality.png", bbox_inches="tight")
QS.to_csv(f"{FIG}/F7_lltc_locality.csv", index=False)

# F8: eigenvalues of P_theta along a 1-D error slice
sl = np.linspace(-1.0, 1.0, 81)
EV = []
for v in sl:
    ee = jnp.zeros((1, X.NE)).at[0, 0].set(v)
    EV.append(np.linalg.eigvalsh(np.asarray(M.lltc_matrix(params, ee, EPS))[0]))
EV = np.array(EV)
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for i in range(X.NE):
    ax.semilogy(sl, np.maximum(EV[:, i], 1e-12), lw=1.1,
                marker=["o", "s", "^", "v", "D", "P", "X", "*", "<"][i],
                markevery=12, ms=3.5, label=f"$\\lambda_{{{i}}}$")
ax.set_xlabel("$e_x$ [m]"); ax.set_ylabel(r"eig $P_\theta(e)$ (log)")
ax.set_title(r"F8  spectrum of $P_\theta$ along an error slice")
ax.legend(fontsize=6, ncol=3)
fig.savefig(f"{FIG}/F8_lltc_spectrum.png", bbox_inches="tight")
pd.DataFrame(EV, index=sl).to_csv(f"{FIG}/F8_lltc_spectrum.csv")
print(f"\n  min eigenvalue over the slice = {EV.min():.3e} "
      f"(floor eps = {EPS:g}): PSD holds by construction, everywhere.")

# %% [markdown]
# ## What this notebook establishes
#
# The verdict is whatever the cells above printed, not what was expected.
#
# - The fit diagnostics are in T3. If the gate fired, the horizon-equivalence
#   claim is unsupported and the two levers are `CFG['n_cand']` (acceptance
#   binds) and `CFG['disp']` against the printed terminal-set reach.
# - The horizon-equivalence verdict is printed in step 3, from the ratio column.
# - **Independently of the outcome**, the structural limitation stands:
#   $\mathbf P_\theta$ is fitted where the data is. Out of distribution it
#   carries no positive-definiteness guarantee beyond the $\varepsilon I$ floor
#   (F8 shows the floor doing its job) and no dissipativity guarantee at all.
# - And LLTC leaves all 13 stage weights untouched. That is why Notebook 3
#   exists.
#
# `lltc/model.pkl` holds a **terminal cost network, not an actor**. Notebooks 6
# and 7 rebuild it through `study_moderate.make_lltc_ctrl`, never through a
# generic loader keyed on `'actor'` — such a loader raises `KeyError` and
# silently drops the row.

# %%
S.checkpoint(model=("lltc", "model.pkl"), fit=("lltc", "nb2_fit.csv"),
             equivalence=("lltc", "nb2_horizon_equivalence.csv"),
             ood=("lltc", "nb2_ood.csv"))
print("\nNotebook 2 complete.")
