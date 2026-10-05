#!/usr/bin/env python3
"""
LLTC-NMPC (Learned Lyapunov Terminal Cost, N = 1) vs long-horizon NMPC
on quadrotor *hover stabilisation* with CTBR actions.

Method (Abdufattokhov, Zanon, Bemporad, IJRNC 2024)
---------------------------------------------------
Baseline:  NMPC with horizon N, quadratic terminal cost F(x) = ||x-xr||^2_P
           (P from the DARE of the hover linearisation), no terminal
           constraint  -> "CDA-NMPC", problem P_N(p, X).
LLTC:      the cost-to-go V(x1) of that problem is replaced by a learned
           quadratic  V(e) = e' Phat(e0) e,  Phat = L L' + eps I,  L lower
           triangular, produced by a ReLU feedforward net.  The horizon then
           collapses to N = 1 (4 inputs + 9 states of decision variables).
Learning:  MSE on V1 = J_N - l0 plus l1-penalties on the two Lyapunov/
           invariance conditions of the paper, eq. (26b)-(26c):
               (dom)  V(e1) <= C_N,        C_N = (N-1) d + lT
               (dec)  V(e1) - V(e0) + l0 <= 0

Hover stabilisation only => the reference is CONSTANT, so p = e0 = x - xr and
the terminal cost is *not* time-varying: Phat depends on the current state
only, is evaluated once per step outside the NLP, and enters it as a parameter.

Model (mass-normalised, CTBR, no disturbances)
----------------------------------------------
    x = [px py pz  vx vy vz  phi theta psi]      (9)
    u = [c wx wy wz]                             (4)   c = thrust/mass [m/s^2]
    pdot = v
    vdot = c * [ cphi sth cpsi + sphi spsi ,
                 cphi sth spsi - sphi cpsi ,
                 cphi cth ] - [0,0,g]
    [phidot thetadot psidot] = T(phi,theta) * omega          (Euler kinematics)
RK4, sampling time TS.  No wind, no drag, no model error: the two controllers
see exactly the simulator.

Outputs (artifacts/lltc_hover/)
-------------------------------
    fig1_fit_quality.png     R^2 of the learned cost-to-go (train/test)
    fig2_altitude.png        altitude tracking, LLTC vs NMPC
    fig3_control_effort.png  control effort, LLTC vs NMPC
    fig4_ood.png             out-of-distribution initial states, LLTC vs NMPC
    fig5_weight_sensitivity.png   sensitivity to weights, LLTC alone
    fig6_computation.png     computational complexity, LLTC vs NMPC
    *.csv                    the numbers behind every figure

Run:  python lltc/lltc_hover.py            (full,  ~15 min)
      python lltc/lltc_hover.py --quick    (smoke, ~2 min, NOT results)
"""
from __future__ import annotations

import argparse
import os
import time

import casadi as ca
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

# ----------------------------------------------------------------------------
# 1. configuration  --  the horizon of the baseline NMPC is set HERE
# ----------------------------------------------------------------------------
N_MPC = 25            # <<< prediction horizon N of the baseline NMPC
TS    = 0.05          # sampling time [s]  (20 Hz CTBR)
G     = 9.81
NX, NU = 9, 4

Z_REF = 1.0                                     # hover altitude [m]
XR = np.array([0., 0., Z_REF, 0., 0., 0., 0., 0., 0.])
UR = np.array([G, 0., 0., 0.])                  # hover input

# stage cost  l = ||x-xr||^2_Qx + ||u-ur||^2_Qu
QX = np.diag([12., 12., 12.,  2., 2., 2.,  1.5, 1.5, 0.2])
QU = np.diag([0.5, 0.05, 0.05, 0.02])

# box constraints  X, U
XLB = np.array([-5., -5., -4., -6., -6., -6., -0.7, -0.7, -np.pi])
XUB = np.array([ 5.,  5.,  6.,  6.,  6.,  6.,  0.7,  0.7,  np.pi])
ULB = np.array([2.0, -4., -4., -4.])            # thrust/weight in [0.2, 1.8]
UUB = np.array([18.0, 4.,  4.,  4.])

# sampling box for the learning data set (pos +-16 cm, vel +-0.5 m/s, att +-7 deg):
#   e0 = x0 - xr = s * U(-BOX, BOX),   s ~ U(0, 1).
# The radial factor s matters: a plain uniform draw in 9-D almost never lands
# near hover, which is exactly where the closed loop spends its time (measured:
# test R^2 0.937 -> 0.985, LLTC tail error 1 mm -> 10 um).  Algorithm 1 then
# keeps only the draws with x0 in Omega_N (J_N <= l0 + C_N).
BOX = np.array([0.16, 0.16, 0.16, 0.48, 0.48, 0.40, 0.12, 0.12, 0.24])

KAPPA  = 3.0          # terminal cost P = KAPPA * P_dare.  The DARE solution
                      # satisfies the decrease condition (7) with EQUALITY for
                      # the linearisation (Bellman), so kappa > 1 buys the
                      # strict margin (kappa-1)*l that absorbs the nonlinearity.
                      # At kappa = 3 the valid ellipsoid stops growing (it is
                      # then limited by the body-rate bounds, not by (7)).
EPS_P  = 1e-2         # Phat = L L' + EPS_P I
NSIM   = 120          # closed-loop steps (6 s)
LAM    = 10.0         # nominal Lyapunov penalty weight (lambda1 = lambda2)
OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "artifacts", "lltc_hover")

torch.set_default_dtype(torch.float64)

# ----------------------------------------------------------------------------
# 2. model
# ----------------------------------------------------------------------------
def f_cont(x, u):
    """Continuous-time CTBR quadrotor dynamics (CasADi-compatible)."""
    v = x[3:6]
    phi, th, psi = x[6], x[7], x[8]
    c, wx, wy, wz = u[0], u[1], u[2], u[3]
    sphi, cphi = ca.sin(phi), ca.cos(phi)
    sth,  cth  = ca.sin(th),  ca.cos(th)
    spsi, cpsi = ca.sin(psi), ca.cos(psi)
    acc = ca.vertcat(c * (cphi * sth * cpsi + sphi * spsi),
                     c * (cphi * sth * spsi - sphi * cpsi),
                     c * cphi * cth - G)
    eul = ca.vertcat(wx + sphi * sth / cth * wy + cphi * sth / cth * wz,
                     cphi * wy - sphi * wz,
                     (sphi * wy + cphi * wz) / cth)
    return ca.vertcat(v, acc, eul)


def f_disc(x, u):
    """One RK4 step of length TS."""
    k1 = f_cont(x, u)
    k2 = f_cont(x + TS / 2 * k1, u)
    k3 = f_cont(x + TS / 2 * k2, u)
    k4 = f_cont(x + TS * k3, u)
    return x + TS / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


_x, _u = ca.SX.sym("x", NX), ca.SX.sym("u", NU)
F_STEP = ca.Function("F_STEP", [_x, _u], [f_disc(_x, _u)])
A_HOV = np.array(ca.Function("A", [_x, _u], [ca.jacobian(f_disc(_x, _u), _x)])(XR, UR))
B_HOV = np.array(ca.Function("B", [_x, _u], [ca.jacobian(f_disc(_x, _u), _u)])(XR, UR))


def stage(x, u, Qx=QX, Qu=QU):
    e, du = np.asarray(x) - XR, np.asarray(u) - UR
    return float(e @ Qx @ e + du @ Qu @ du)


# ----------------------------------------------------------------------------
# 3. terminal cost, terminal set, and the invariance level C_N
# ----------------------------------------------------------------------------
def sym_inv_sqrt(M):
    w, V = np.linalg.eigh(M)
    return V @ np.diag(w ** -0.5) @ V.T


def terminal_ingredients(N, rng, n_probe=400):
    """P from the DARE; largest ellipsoid level lT on which Assumption 4 holds
    for u_T = ur + K(x-xr); d = inf_{x notin X_T} l ; C_N = (N-1)d + lT."""
    Pl = solve_discrete_are(A_HOV, B_HOV, QX, QU)
    K = -np.linalg.solve(QU + B_HOV.T @ Pl @ B_HOV, B_HOV.T @ Pl @ A_HOV)
    P = KAPPA * Pl
    Pinv_sqrt = sym_inv_sqrt(P)

    def feasible(alpha):
        y = rng.normal(size=(n_probe, NX))
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        Z = (Pinv_sqrt @ (np.sqrt(alpha) * y).T).T          # z' P z = alpha
        for z in Z:
            x, u = XR + z, UR + K @ z
            if np.any(x < XLB) or np.any(x > XUB):
                return False
            if np.any(u < ULB) or np.any(u > UUB):
                return False
            xn = np.array(F_STEP(x, u)).ravel()
            en = xn - XR
            if en @ P @ en - z @ P @ z + stage(x, u) > 0:    # eq. (7)
                return False
        return True

    lT = None
    for alpha in np.logspace(3, -3, 61):
        if feasible(alpha):
            lT = float(alpha)
            break
    if lT is None:
        raise RuntimeError("no valid terminal level found")
    d = lT * float(np.min(np.linalg.eigvalsh(Pinv_sqrt @ QX @ Pinv_sqrt)))
    return P, K, lT, d, (N - 1) * d + lT


# ----------------------------------------------------------------------------
# 4. NMPC problems (CasADi + IPOPT), built once, solved as parametric NLPs
# ----------------------------------------------------------------------------
def build_nmpc(N, Qx, Qu, Pterm=None, P_as_param=False, max_iter=300):
    """Multiple-shooting P_N(p, X): terminal cost, no terminal constraint.
    P_as_param=True leaves the terminal Hessian as an NLP parameter (LLTC)."""
    Xv, Uv = ca.SX.sym("X", NX, N + 1), ca.SX.sym("U", NU, N)
    x0p = ca.SX.sym("x0p", NX)
    par = [x0p]
    if P_as_param:
        Pp = ca.SX.sym("Pp", NX, NX)
        par.append(ca.vec(Pp))
        Pt = Pp
    else:
        Pt = ca.DM(Pterm)

    J, g = 0, [Xv[:, 0] - x0p]
    for k in range(N):
        e, du = Xv[:, k] - XR, Uv[:, k] - UR
        J += ca.bilin(ca.DM(Qx), e, e) + ca.bilin(ca.DM(Qu), du, du)
        g.append(Xv[:, k + 1] - f_disc(Xv[:, k], Uv[:, k]))
    eN = Xv[:, N] - XR
    J += ca.bilin(Pt, eN, eN)

    w = ca.vertcat(ca.vec(Uv), ca.vec(Xv))
    nlp = dict(x=w, f=J, g=ca.vertcat(*g), p=ca.vertcat(*par))
    opts = {"print_time": 0, "ipopt.print_level": 0, "ipopt.sb": "yes",
            "ipopt.max_iter": max_iter, "ipopt.tol": 1e-7,
            "ipopt.acceptable_tol": 1e-5}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    lbw = np.concatenate([np.tile(ULB, N), np.tile(XLB, N + 1)])
    ubw = np.concatenate([np.tile(UUB, N), np.tile(XUB, N + 1)])
    ng = (N + 1) * NX

    def solve(x0, Pmat=None, w0=None):
        p = np.concatenate([x0] if not P_as_param
                           else [x0, np.asarray(Pmat).reshape(-1, order="F")])
        if w0 is None:
            w0 = np.concatenate([np.tile(UR, N), np.tile(x0, N + 1)])
        r = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=np.zeros(ng), ubg=np.zeros(ng), p=p)
        wo = np.array(r["x"]).ravel()
        U = wo[:NU * N].reshape(NU, N, order="F")
        X = wo[NU * N:].reshape(NX, N + 1, order="F")
        return dict(u0=U[:, 0], x1=X[:, 1], J=float(r["f"]), w=wo,
                    ok=bool(solver.stats()["success"]), U=U, X=X)

    solve.n_var, solve.n_con = int(w.shape[0]), ng
    return solve


# ----------------------------------------------------------------------------
# 5. data collection  (Algorithm 1: keep samples with x0 in Omega_N)
# ----------------------------------------------------------------------------
def collect(nmpc, M, CN, rng, box=BOX, max_trials=30):
    keys = ("e0", "e1", "l0", "V1")
    D = {k: [] for k in keys}
    trials = accepted = 0
    while len(D["V1"]) < M and trials < max_trials * M:
        trials += 1
        e0 = rng.uniform() * rng.uniform(-box, box)
        x0 = np.clip(XR + e0, XLB, XUB)
        r = nmpc(x0)
        if not r["ok"]:
            continue
        l0 = stage(x0, r["u0"])
        V1 = r["J"] - l0                                  # cost-to-go, eq. (19)
        if r["J"] > l0 + CN:                              # x0 in Omega_N ?
            continue
        accepted += 1
        D["e0"].append(x0 - XR)
        D["e1"].append(r["x1"] - XR)
        D["l0"].append(l0)
        D["V1"].append(V1)
    D = {k: np.array(v) for k, v in D.items()}
    D["rate"] = accepted / max(trials, 1)
    return D


# ----------------------------------------------------------------------------
# 6. the learned Lyapunov terminal cost
# ----------------------------------------------------------------------------
TRIL = np.tril_indices(NX)


class LNet(nn.Module):
    """p -> vec(L(p)); Phat = L L' + eps I  (positive definite by construction)."""

    def __init__(self, hid=64, layers=3):
        super().__init__()
        seq, d = [], NX
        for _ in range(layers):
            seq += [nn.Linear(d, hid), nn.ReLU()]
            d = hid
        seq += [nn.Linear(d, len(TRIL[0]))]
        self.net = nn.Sequential(*seq)
        self.register_buffer("scale", torch.as_tensor(BOX))

    def P(self, e0):
        v = self.net(e0 / self.scale)
        L = torch.zeros(v.shape[0], NX, NX, dtype=v.dtype)
        L[:, TRIL[0], TRIL[1]] = v
        return L @ L.transpose(1, 2) + EPS_P * torch.eye(NX, dtype=v.dtype)

    def V(self, e0, e):
        """V(e; p=e0) = e' Phat(e0) e."""
        return torch.einsum("bi,bij,bj->b", e, self.P(e0), e)

    @torch.no_grad()
    def P_np(self, e0):
        return self.P(torch.as_tensor(e0)[None]).numpy()[0]


def train(D, CN, lam1=LAM, lam2=LAM, epochs=4000, lr=1e-3, seed=0, split=0.8,
          hid=64, layers=3, verbose=True):
    torch.manual_seed(seed)
    n = len(D["V1"])
    idx = np.random.default_rng(seed).permutation(n)
    tr, te = idx[:int(split * n)], idx[int(split * n):]
    T = {k: torch.as_tensor(D[k]) for k in ("e0", "e1", "l0", "V1")}
    s = float(T["V1"].mean())                         # loss scaling only

    net = LNet(hid, layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        opt.zero_grad()
        V1h = net.V(T["e0"][tr], T["e1"][tr])
        V0h = net.V(T["e0"][tr], T["e0"][tr])
        fit = ((V1h - T["V1"][tr]) ** 2).mean() / s ** 2
        dom = torch.relu(V1h - CN).mean() / s                       # (26b)
        dec = torch.relu(V1h - V0h + T["l0"][tr]).mean() / s        # (26c)
        (fit + lam1 * dom + lam2 * dec).backward()
        opt.step()
        sch.step()
        if verbose and (ep % 1000 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:5d}  fit {fit.item():.2e}  "
                  f"dom {dom.item():.2e}  dec {dec.item():.2e}")

    with torch.no_grad():
        V1h = net.V(T["e0"], T["e1"]).numpy()
        V0h = net.V(T["e0"], T["e0"]).numpy()
    V1 = D["V1"]

    def r2(i):
        return 1 - ((V1h[i] - V1[i]) ** 2).sum() / ((V1[i] - V1[i].mean()) ** 2).sum()

    def nrmse(i):
        return float(np.sqrt(((V1h[i] - V1[i]) ** 2).mean()) / V1[i].std())

    m = dict(R2_train=float(r2(tr)), R2_test=float(r2(te)),
             nrmse_train=nrmse(tr), nrmse_test=nrmse(te),
             C_dom_train=int((V1h[tr] > CN).sum()), C_dom_test=int((V1h[te] > CN).sum()),
             C_dec_train=int((V1h[tr] - V0h[tr] + D["l0"][tr] > 0).sum()),
             C_dec_test=int((V1h[te] - V0h[te] + D["l0"][te] > 0).sum()),
             n_train=len(tr), n_test=len(te))
    return net, m, (tr, te), V1h


# ----------------------------------------------------------------------------
# 7. closed loop
# ----------------------------------------------------------------------------
def simulate(ctrl, x0, nsim=None):
    nsim = NSIM if nsim is None else nsim
    x, Xs, Us, ts, cost = x0.copy(), [x0.copy()], [], [], 0.0
    warm = None
    for _ in range(nsim):
        t0 = time.perf_counter()
        u, warm = ctrl(x, warm)
        ts.append((time.perf_counter() - t0) * 1e3)
        cost += stage(x, u)                            # nominal-weight cost
        x = np.array(F_STEP(x, np.clip(u, ULB, UUB))).ravel()
        Xs.append(x.copy())
        Us.append(u)
    Xs, Us = np.array(Xs), np.array(Us)
    return dict(X=Xs, U=Us, t=np.array(ts), cost=cost,
                effort=float(np.sum([(u - UR) @ QU @ (u - UR) for u in Us])),
                final_err=float(np.linalg.norm(Xs[-1, :3] - XR[:3])))


def make_ctrl(solve, net=None):
    if net is None:
        def ctrl(x, warm):
            r = solve(x, w0=warm)
            return r["u0"], r["w"]
    else:
        def ctrl(x, warm):
            P = net.P_np(x - XR)                       # FNN evaluation, timed
            r = solve(x, Pmat=P, w0=warm)
            return r["u0"], r["w"]
    return ctrl


# ----------------------------------------------------------------------------
# 8. experiments + figures
# ----------------------------------------------------------------------------
def plain_log(ax, axis="y"):
    """Log axis with plain-number labels and no minor-tick labels."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(FormatStrFormatter("%g"))
    a.set_minor_formatter(NullFormatter())


def savefig(fig, name):
    p = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("  wrote", os.path.relpath(p))


def savecsv(name, header, rows):
    p = os.path.join(OUT, name)
    with open(p, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(f"{v:.6g}" if isinstance(v, (int, float)) else str(v)
                              for v in r) + "\n")
    print("  wrote", os.path.relpath(p))


def fig_fit(D, V1h, split, m):
    tr, te = split
    V1 = D["V1"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    lim = [0, 1.05 * max(V1.max(), V1h.max())]
    ax[0].plot(lim, lim, "k--", lw=1, label="ideal")
    ax[0].scatter(V1[tr], V1h[tr], s=8, alpha=.45,
                  label=f"train  $R^2$={m['R2_train']:.4f}")
    ax[0].scatter(V1[te], V1h[te], s=12, alpha=.75, marker="^",
                  label=f"test   $R^2$={m['R2_test']:.4f}")
    ax[0].set(xlabel=r"true cost-to-go $V_1=J_N-\ell_0$",
              ylabel=r"learned $\hat V_{LLTC}(x_1)$",
              title="fit quality of the cost-to-go", xlim=lim, ylim=lim)
    ax[0].legend(loc="upper left")
    ne = np.linalg.norm(D["e0"] / BOX, axis=1)
    rel = (V1h - V1) / V1
    ax[1].scatter(ne[tr], 100 * rel[tr], s=8, alpha=.45, label="train")
    ax[1].scatter(ne[te], 100 * rel[te], s=12, alpha=.75, marker="^", label="test")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set(xlabel=r"normalised $\|e_0\|$", ylabel="relative error [%]",
              title=f"NRMSE {m['nrmse_train']:.3f}/{m['nrmse_test']:.3f} "
                    f"(train/test)")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=.3)
    savefig(fig, "fig1_fit_quality.png")
    savecsv("fit.csv", ["V1_true", "V1_pred", "split"],
            [(V1[i], V1h[i], "train" if i in set(tr.tolist()) else "test")
             for i in range(len(V1))])


def fig_tracking(runs, ics):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = plt.cm.viridis(np.linspace(0, .85, len(ics)))
    t = np.arange(runs["nmpc"][0]["X"].shape[0]) * TS
    for i, c in enumerate(cols):
        ax[0].plot(t, runs["nmpc"][i]["X"][:, 2], color=c, lw=1.8,
                   label=f"NMPC $N$={N_MPC}" if i == 0 else None)
        ax[0].plot(t, runs["lltc"][i]["X"][:, 2], color=c, lw=1.6, ls="--",
                   label="LLTC $N$=1" if i == 0 else None)
        ax[1].plot(t, np.linalg.norm(runs["nmpc"][i]["X"][:, :3] - XR[:3], axis=1),
                   color=c, lw=1.8)
        ax[1].plot(t, np.linalg.norm(runs["lltc"][i]["X"][:, :3] - XR[:3], axis=1),
                   color=c, lw=1.6, ls="--")
    ax[0].axhline(Z_REF, color="k", lw=1, ls=":", label="reference")
    ax[0].set(xlabel="time [s]", ylabel="altitude $p_z$ [m]",
              title=f"altitude tracking, {len(ics)} initial states")
    ax[0].legend()
    ax[1].set(xlabel="time [s]", ylabel=r"$\|p-p_r\|$ [m]", yscale="log",
              title="position error (solid NMPC, dashed LLTC)")
    for a in ax:
        a.grid(alpha=.3)
    savefig(fig, "fig2_altitude.png")


def fig_effort(runs, ics):
    n = runs["nmpc"][0]["U"].shape[0]
    tu = np.arange(n) * TS
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax[0].plot(tu, runs["nmpc"][0]["U"][:, 0], lw=1.8, label=f"NMPC $N$={N_MPC}")
    ax[0].plot(tu, runs["lltc"][0]["U"][:, 0], lw=1.6, ls="--", label="LLTC $N$=1")
    ax[0].axhline(G, color="k", ls=":", lw=1, label="hover $g$")
    ax[0].set(xlabel="time [s]", ylabel="collective thrust $c$ [m/s$^2$]",
              title="CTBR thrust, first initial state")
    ax[0].legend()
    for k, (lab, ls) in enumerate([("NMPC", "-"), ("LLTC", "--")]):
        key = "nmpc" if k == 0 else "lltc"
        U = runs[key][0]["U"]
        ax[1].plot(tu, np.linalg.norm(U[:, 1:], axis=1), ls=ls, lw=1.7,
                   label=f"{lab}  $\\|\\omega\\|$")
    ax[1].set(xlabel="time [s]", ylabel=r"$\|\omega\|$ [rad/s]",
              title="body-rate command magnitude")
    ax[1].legend()
    w = 0.38
    x = np.arange(len(ics))
    en = [r["effort"] for r in runs["nmpc"]]
    el = [r["effort"] for r in runs["lltc"]]
    ax[2].bar(x - w / 2, en, w, label=f"NMPC $N$={N_MPC}")
    ax[2].bar(x + w / 2, el, w, label="LLTC $N$=1")
    for xi, (a, b) in enumerate(zip(en, el)):
        ax[2].text(xi, max(a, b) * 1.02, f"{100*(b-a)/a:+.1f}%", ha="center",
                   fontsize=8)
    ax[2].set(xlabel="initial state", xticks=x,
              ylabel=r"$\sum_t\|u_t-u_r\|^2_{Q_u}$",
              title="total control effort")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=.3)
    savefig(fig, "fig3_control_effort.png")
    savecsv("effort.csv", ["ic", "effort_nmpc", "effort_lltc", "cost_nmpc",
                           "cost_lltc"],
            [(i, en[i], el[i], runs["nmpc"][i]["cost"], runs["lltc"][i]["cost"])
             for i in range(len(ics))])


def fig_ood(scales, res):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    k = len(scales) - 1
    t = np.arange(res[k]["nmpc"][0]["X"].shape[0]) * TS
    for j, r in enumerate(res[k]["nmpc"]):
        ax[0].plot(t, r["X"][:, 2], "C0", lw=1.6, alpha=.8,
                   label=f"NMPC $N$={N_MPC}" if j == 0 else None)
    for j, r in enumerate(res[k]["lltc"]):
        ax[0].plot(t, r["X"][:, 2], "C1--", lw=1.5, alpha=.8,
                   label="LLTC $N$=1" if j == 0 else None)
    ax[0].axhline(Z_REF, color="k", ls=":", lw=1)
    ax[0].set(xlabel="time [s]", ylabel="altitude $p_z$ [m]",
              title=f"altitude, {scales[k]:.1f}x training box (OOD)")
    ax[0].legend()
    for key, col, mk in (("nmpc", "C0", "o"), ("lltc", "C1", "s")):
        c = np.array([[r["cost"] for r in res[i][key]] for i in range(len(scales))])
        ax[1].errorbar(scales, c.mean(1), yerr=c.std(1), marker=mk, color=col,
                       capsize=3, label="NMPC" if key == "nmpc" else "LLTC")
        e = np.array([[r["final_err"] for r in res[i][key]]
                      for i in range(len(scales))])
        ax[2].errorbar(scales, e.mean(1), yerr=e.std(1), marker=mk, color=col,
                       capsize=3, label="NMPC" if key == "nmpc" else "LLTC")
    for a in ax[1:]:
        a.axvline(1.0, color="green", ls=":", lw=1.2)
        a.text(1.04, 0.03, "training box", color="green", fontsize=8,
               transform=a.get_xaxis_transform(), va="bottom")
    ax[1].set(xlabel=r"initial-state scale $\times$ training box",
              ylabel="closed-loop cost", yscale="log",
              title="cost vs distance from training data")
    ax[1].legend()
    ax[2].set(xlabel=r"initial-state scale $\times$ training box",
              ylabel=r"final $\|p-p_r\|$ [m]", yscale="log",
              title="terminal accuracy (OOD)")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=.3)
    savefig(fig, "fig4_ood.png")
    savecsv("ood.csv", ["scale", "cost_nmpc", "cost_lltc", "err_nmpc", "err_lltc"],
            [(s,
              np.mean([r["cost"] for r in res[i]["nmpc"]]),
              np.mean([r["cost"] for r in res[i]["lltc"]]),
              np.mean([r["final_err"] for r in res[i]["nmpc"]]),
              np.mean([r["final_err"] for r in res[i]["lltc"]]))
             for i, s in enumerate(scales)])


def fig_weights(lams, lam_rows, wscales, wrows, cost_nominal):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    xl = [max(l, 1e-2) for l in lams]                 # lambda=0 plotted at 1e-2
    xlab = ["0\n(LTC)" if l == 0 else f"{l:g}" for l in lams]
    r2 = [r["R2_test"] for r in lam_rows]
    ax[0, 0].semilogx(xl, r2, "o-")
    ax[0, 0].set(xlabel=r"Lyapunov penalty $\lambda_1=\lambda_2$",
                 ylabel=r"test $R^2$ of $\hat V$",
                 title=r"fit vs penalty weight ($\lambda=0$ is LTC)")
    ax[0, 1].semilogx(xl, [r["C_dom_test"] + r["C_dom_train"] for r in lam_rows],
                      "o-", label="(26b) domain $\\hat V(x_1)>C_N$")
    ax[0, 1].semilogx(xl, [r["C_dec_test"] + r["C_dec_train"] for r in lam_rows],
                      "s--", label="(26c) decrease violated")
    ax[0, 1].set(xlabel=r"$\lambda_1=\lambda_2$", ylabel="# violating samples",
                 title="Lyapunov conditions on the data set", yscale="symlog")
    ax[0, 1].legend()
    ax[1, 0].loglog(xl, [r["cost"] for r in lam_rows], "o-")
    plain_log(ax[1, 0])
    ax[1, 0].axhline(cost_nominal, color="k", ls=":", lw=1,
                     label=f"$\\lambda$={LAM:g} (nominal)")
    ax[1, 0].set(xlabel=r"$\lambda_1=\lambda_2$",
                 ylabel="mean closed-loop cost (LLTC)",
                 title="closed-loop cost vs penalty weight")
    ax[1, 0].legend()
    for lab, key, mk in (("$Q_x$ scaled (no effect: at $N$=1 $Q_x$ only "
                          "weights the fixed $x_0$)", "qx", "o"),
                         ("$Q_u$ scaled", "qu", "s")):
        ax[1, 1].semilogx(wscales, [r[key] for r in wrows], marker=mk, label=lab)
    ax[1, 1].axhline(cost_nominal, color="k", ls=":", lw=1, label="nominal weights")
    ax[1, 1].set_xticks(wscales, [f"{w:g}" for w in wscales])
    ax[1, 1].xaxis.set_minor_formatter(NullFormatter())
    ax[1, 1].set(xlabel=r"stage-weight scaling used online",
                 ylabel="closed-loop cost (nominal metric)",
                 title="stale terminal cost under re-tuned weights")
    ax[1, 1].legend(fontsize=8)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_xticks(xl, xlab)
        a.xaxis.set_minor_formatter(NullFormatter())
    for a in ax.ravel():
        a.grid(alpha=.3)
    savefig(fig, "fig5_weight_sensitivity.png")
    savecsv("weight_sensitivity.csv",
            ["kind", "value", "R2_test", "C_dom", "C_dec", "cost"],
            [("lambda", lams[i], r["R2_test"],
              r["C_dom_train"] + r["C_dom_test"],
              r["C_dec_train"] + r["C_dec_test"], r["cost"])
             for i, r in enumerate(lam_rows)]
            + [("Qx_scale", wscales[i], np.nan, np.nan, np.nan, r["qx"])
               for i, r in enumerate(wrows)]
            + [("Qu_scale", wscales[i], np.nan, np.nan, np.nan, r["qu"])
               for i, r in enumerate(wrows)])


def fig_timing(t_nmpc, t_lltc, horizons, t_hor, dims):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    w = 0.35
    for i, (lab, t) in enumerate((("NMPC $N$=%d" % N_MPC, t_nmpc),
                                 ("LLTC $N$=1", t_lltc))):
        ax[0].bar([0 + (i - .5) * w, 1 + (i - .5) * w], [t.mean(), t.max()], w,
                  label=lab)
    for i, t in enumerate((t_nmpc, t_lltc)):
        for j, v in enumerate((t.mean(), t.max())):
            ax[0].text(j + (i - .5) * w, v, f"{v:.1f}", ha="center",
                       va="bottom", fontsize=8)
    ax[0].set(xticks=[0, 1], xticklabels=[r"average $T_a$", r"worst case $T_w$"],
              ylabel="solve time per step [ms]", yscale="log",
              title=f"speed-up  $T_a$: {t_nmpc.mean()/t_lltc.mean():.1f}x,"
                    f"  $T_w$: {t_nmpc.max()/t_lltc.max():.1f}x")
    ax[0].legend()
    bins = np.geomspace(min(t_lltc.min(), t_nmpc.min()),
                        max(t_lltc.max(), t_nmpc.max()), 50)
    ax[1].hist(t_nmpc, bins=bins, alpha=.7, label=f"NMPC $N$={N_MPC}")
    ax[1].hist(t_lltc, bins=bins, alpha=.7, label="LLTC $N$=1")
    ax[1].set(xlabel="solve time per step [ms]", ylabel="# steps", xscale="log",
              title="per-step distribution")
    ax[1].legend()
    ax[2].plot(horizons, t_hor, "o-", label="NMPC vs horizon $N$")
    ax[2].axhline(t_lltc.mean(), color="C1", ls="--",
                  label=f"LLTC $N$=1 ({t_lltc.mean():.2f} ms)")
    ax[2].axvline(N_MPC, color="k", ls=":", lw=1, label=f"baseline $N$={N_MPC}")
    ax[2].set(xlabel="prediction horizon $N$", ylabel="mean solve time [ms]",
              yscale="log", title="complexity vs horizon")
    ax[2].legend()
    plain_log(ax[0])
    plain_log(ax[1], "x")
    plain_log(ax[2])
    for a in ax:
        a.grid(alpha=.3)
    savefig(fig, "fig6_computation.png")
    savecsv("timing.csv", ["controller", "n_var", "n_con", "Ta_ms", "Tw_ms"],
            [(f"NMPC_N{N_MPC}", dims[0][0], dims[0][1], t_nmpc.mean(), t_nmpc.max()),
             ("LLTC_N1", dims[1][0], dims[1][1], t_lltc.mean(), t_lltc.max())])
    savecsv("timing_vs_horizon.csv", ["N", "mean_ms"],
            list(zip(horizons, t_hor)))


# ----------------------------------------------------------------------------
# 9. main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=1500, help="data samples")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n-ic", type=int, default=4, help="closed-loop initial states")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="smoke run, not results")
    a = ap.parse_args()
    if a.quick:
        a.M, a.epochs, a.n_ic = 200, 600, 2
        global NSIM
        NSIM = 60
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    print(f"[1] terminal ingredients (N = {N_MPC})")
    P, K, lT, d, CN = terminal_ingredients(N_MPC, rng)
    print(f"    l_T = {lT:.4g}   d = {d:.4g}   C_N = {CN:.4g}")

    print("[2] building NMPC problems")
    nmpc = build_nmpc(N_MPC, QX, QU, Pterm=P)
    lltc = build_nmpc(1, QX, QU, P_as_param=True)
    print(f"    NMPC: {nmpc.n_var} vars / {nmpc.n_con} eq.   "
          f"LLTC: {lltc.n_var} vars / {lltc.n_con} eq.")

    print(f"[3] collecting M = {a.M} cost-to-go samples")
    t0 = time.time()
    D = collect(nmpc, a.M, CN, rng)
    print(f"    {len(D['V1'])} samples, acceptance {100*D['rate']:.1f}%, "
          f"{time.time()-t0:.0f}s,  V1 in [{D['V1'].min():.3g},{D['V1'].max():.3g}]")

    print(f"[4] training the Lyapunov terminal cost (lambda = {LAM:g})")
    net, m, split, V1h = train(D, CN, epochs=a.epochs, seed=a.seed)
    print(f"    R2 = {m['R2_train']:.4f}/{m['R2_test']:.4f} (train/test), "
          f"NRMSE = {m['nrmse_train']:.3f}/{m['nrmse_test']:.3f}, "
          f"violations dom {m['C_dom_train']}/{m['C_dom_test']} "
          f"dec {m['C_dec_train']}/{m['C_dec_test']}")
    fig_fit(D, V1h, split, m)

    print("[5] closed loop, LLTC vs NMPC")
    ics = [XR + rng.uniform(-BOX, BOX) for _ in range(a.n_ic)]
    runs = {"nmpc": [simulate(make_ctrl(nmpc), x0) for x0 in ics],
            "lltc": [simulate(make_ctrl(lltc, net), x0) for x0 in ics]}
    for i in range(a.n_ic):
        print(f"    ic{i}: cost {runs['nmpc'][i]['cost']:8.2f} (NMPC) vs "
              f"{runs['lltc'][i]['cost']:8.2f} (LLTC)   final err "
              f"{runs['nmpc'][i]['final_err']:.3f} / "
              f"{runs['lltc'][i]['final_err']:.3f} m")
    fig_tracking(runs, ics)
    fig_effort(runs, ics)

    print("[6] out-of-distribution initial states")
    scales = [1.0, 1.5, 2.0, 3.0, 4.0] if not a.quick else [1.0, 2.0]
    n_ood = 3 if not a.quick else 2
    dirs = [rng.uniform(-1, 1, NX) for _ in range(n_ood)]
    res = []
    for s in scales:
        xs = [np.clip(XR + s * dv * BOX, XLB + 1e-3, XUB - 1e-3) for dv in dirs]
        res.append({"nmpc": [simulate(make_ctrl(nmpc), x0) for x0 in xs],
                    "lltc": [simulate(make_ctrl(lltc, net), x0) for x0 in xs]})
        print(f"    {s:.1f}x box: cost {np.mean([r['cost'] for r in res[-1]['nmpc']]):.1f}"
              f" (NMPC) vs {np.mean([r['cost'] for r in res[-1]['lltc']]):.1f} (LLTC)")
    fig_ood(scales, res)

    print("[7] sensitivity to weights (LLTC alone)")
    lams = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0] if not a.quick else [0.0, 10.0]
    ep_s = max(a.epochs // 2, 400)
    ic_s = ics[:min(3, a.n_ic)]
    lam_rows = []
    for lam in lams:
        net_l, m_l, _, _ = train(D, CN, lam1=lam, lam2=lam, epochs=ep_s,
                                 seed=a.seed, verbose=False)
        c = np.mean([simulate(make_ctrl(lltc, net_l), x0)["cost"] for x0 in ic_s])
        m_l["cost"] = float(c)
        lam_rows.append(m_l)
        print(f"    lambda {lam:7g}: R2_test {m_l['R2_test']:.4f}  "
              f"viol {m_l['C_dom_train']+m_l['C_dom_test']}/"
              f"{m_l['C_dec_train']+m_l['C_dec_test']}  cost {c:.2f}")
    cost_nom = float(np.mean([r["cost"] for r in runs["lltc"][:len(ic_s)]]))
    wscales = [0.25, 0.5, 1.0, 2.0, 4.0] if not a.quick else [1.0, 2.0]
    wrows = []
    for s in wscales:
        row = {}
        for key, Qx, Qu in (("qx", s * QX, QU), ("qu", QX, s * QU)):
            sv = build_nmpc(1, Qx, Qu, P_as_param=True)
            row[key] = float(np.mean([simulate(make_ctrl(sv, net), x0)["cost"]
                                      for x0 in ic_s]))
        wrows.append(row)
        print(f"    weight scale {s:5g}: cost(Qx) {row['qx']:.2f}  "
              f"cost(Qu) {row['qu']:.2f}")
    fig_weights(lams, lam_rows, wscales, wrows, cost_nom)

    print("[8] computational complexity")
    t_nmpc = np.concatenate([r["t"] for r in runs["nmpc"]])
    t_lltc = np.concatenate([r["t"] for r in runs["lltc"]])
    horizons = [1, 5, 10, 15, 20, 25, 30, 40] if not a.quick else [1, 10, 25]
    t_hor = []
    for N in horizons:
        sv = build_nmpc(N, QX, QU, Pterm=P)
        r = simulate(make_ctrl(sv), ics[0], nsim=min(40, NSIM))
        t_hor.append(float(r["t"].mean()))
        print(f"    N = {N:3d}: {t_hor[-1]:7.2f} ms/step")
    fig_timing(t_nmpc, t_lltc, horizons, t_hor, [(nmpc.n_var, nmpc.n_con),
                                                 (lltc.n_var, lltc.n_con)])

    savecsv("summary.csv", ["quantity", "value"],
            [("N_baseline", N_MPC), ("Ts", TS), ("l_T", lT), ("d", d), ("C_N", CN),
             ("M_samples", len(D["V1"])), ("acceptance_rate", D["rate"]),
             ("R2_train", m["R2_train"]), ("R2_test", m["R2_test"]),
             ("NRMSE_test", m["nrmse_test"]),
             ("C_dom_total", m["C_dom_train"] + m["C_dom_test"]),
             ("C_dec_total", m["C_dec_train"] + m["C_dec_test"]),
             ("cost_nmpc", float(np.mean([r["cost"] for r in runs["nmpc"]]))),
             ("cost_lltc", float(np.mean([r["cost"] for r in runs["lltc"]]))),
             ("effort_nmpc", float(np.mean([r["effort"] for r in runs["nmpc"]]))),
             ("effort_lltc", float(np.mean([r["effort"] for r in runs["lltc"]]))),
             ("Ta_nmpc_ms", float(t_nmpc.mean())), ("Ta_lltc_ms", float(t_lltc.mean())),
             ("Tw_nmpc_ms", float(t_nmpc.max())), ("Tw_lltc_ms", float(t_lltc.max())),
             ("speedup_Ta", float(t_nmpc.mean() / t_lltc.mean())),
             ("speedup_Tw", float(t_nmpc.max() / t_lltc.max()))])
    print("\ndone ->", os.path.relpath(OUT))


if __name__ == "__main__":
    main()
