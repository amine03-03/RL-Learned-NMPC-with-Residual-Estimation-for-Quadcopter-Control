# LLTC-NMPC vs NMPC: quadrotor hover stabilisation

`lltc_hover.py` is one self-contained script, with no notebook, no JAX and no
ROS. It re-implements *Abdufattokhov, Zanon & Bemporad, "Learning Lyapunov
terminal costs from data for complexity reduction in NMPC", IJRNC 2024* on a
quadrotor that holds hover. It compares a one-step NMPC with a learned
Lyapunov terminal cost (**LLTC, N = 1**) against the long-horizon NMPC it was
learned from (**N = `N_MPC`**).

```bash
# from the repository root, with .venv-lltc active (see the main README §2.4)
python lltc/lltc_hover.py            # full run, ~15 min on a laptop CPU
python lltc/lltc_hover.py --quick    # smoke run, ~2 min, numbers are NOT results
```

| option | default | meaning |
|---|---|---|
| `--M` | 1500 | number of cost-to-go samples (Algorithm 1) |
| `--epochs` | 4000 | Adam epochs for the terminal-cost network |
| `--n-ic` | 4 | closed-loop initial states in figs 2, 3 and 6 |
| `--seed` | 0 | seed for sampling, training and initial states |
| `--quick` | off | M=200, 600 epochs, 2 states, 3 s simulations |

The **NMPC horizon** and all other settings are constants at the top of the
script (section 1, *configuration*):

```python
N_MPC = 25            # <<< prediction horizon N of the baseline NMPC
TS    = 0.05          # sampling time [s]
```

---

## 1. Problem

**Model.** Mass-normalised quadrotor with collective thrust / body rate (CTBR)
inputs, no disturbances, RK4-discretised at `TS = 0.05 s`:

```
x = [p (3), v (3), φ θ ψ]       u = [c, ωx, ωy, ωz]      c = thrust / mass  [m/s²]

ṗ = v
v̇ = c · [cφ sθ cψ + sφ sψ,  cφ sθ sψ − sφ cψ,  cφ cθ] − [0, 0, g]
[φ̇ θ̇ ψ̇] = T(φ, θ) · ω                       (Euler-angle kinematics)
```

**Task.** Hover at `p = (0, 0, 1) m`, so `x_r = [0 0 1 0 … 0]` and
`u_r = [g 0 0 0]`. The reference is constant, which means the learned terminal
cost is **not** time-varying: it depends only on the current state.

**Stage cost.** `ℓ = ‖x − x_r‖²_Qx + ‖u − u_r‖²_Qu` with
`Qx = diag(12,12,12, 2,2,2, 1.5,1.5,0.2)` and `Qu = diag(0.5, 0.05,0.05,0.02)`.

**Constraints.**

| quantity | bound |
|---|---|
| thrust `c` | 2 to 18 m/s² (0.2 to 1.8 thrust-to-weight) |
| body rates | ±4 rad/s |
| roll, pitch | ±0.7 rad |
| position | x, y ±5 m; z −4 to 6 m |
| velocity | ±6 m/s |

## 2. Method (paper section → code)

| step | paper | code |
|---|---|---|
| Terminal cost `F = ‖x−x_r‖²_P`, `P = κ·P_DARE`, local law `u_T = u_r + K(x−x_r)` | Assumption 4 | `terminal_ingredients` |
| Largest level `l_T` on which `F(f(x,u_T)) − F(x) + ℓ ≤ 0` and the constraints hold (scan over α, 400 random boundary points each) | eq. (7) | `terminal_ingredients` |
| `d = inf_{x ∉ X_T} ℓ = l_T · λ_min(P^{-½} Qx P^{-½})` (closed form for an ellipsoid) | eq. (14) | `terminal_ingredients` |
| `C_N = (N−1)·d + l_T` | eq. (13) | `terminal_ingredients` |
| Baseline: NMPC with terminal cost and no terminal constraint (multiple shooting, IPOPT) | P_N(p, X), eq. (8) | `build_nmpc` |
| Data: solve P_N, keep `x0` only if `J_N ≤ ℓ0 + C_N` (x0 ∈ Ω_N); store `e0, e1, ℓ0, V1 = J_N − ℓ0` | Algorithm 1 | `collect` |
| `V̂(e1; e0) = e1ᵀ P̂(e0) e1`, `P̂ = L Lᵀ + εI`, L lower-triangular from a 3×64 ReLU net | eqs. (20)–(25) | `LNet` |
| Loss: MSE on V1 + λ1·max(V̂(e1) − C_N, 0) + λ2·max(V̂(e1) − V̂(e0) + ℓ0, 0) | eq. (27) | `train` |
| LLTC controller: evaluate P̂(x_t − x_r), then solve the one-step NLP with P̂ as a parameter | eq. (28), Algorithm 2 | `build_nmpc(1, P_as_param=True)`, `make_ctrl` |

Two implementation choices are measured, not assumed:

- **κ = 3.** The DARE matrix satisfies the decrease condition (7) with
  *equality* for the linearisation (Bellman), so the nonlinear residual breaks
  it immediately (no valid level at κ = 1). With κ > 1 the condition has a
  strict margin `(κ−1)ℓ`. At κ = 3 the valid ellipsoid stops growing, because
  it is then limited by the body-rate bounds rather than by (7).
- **Radial sampling** `e0 = s · U(−BOX, BOX)` with `s ~ U(0,1)`. A plain uniform
  draw in 9-D almost never lands near hover, which is where the closed loop
  spends its time. With the radial factor, test R² rises from 0.937 to 0.985,
  and LLTC's residual position error at 6 s falls from ≈1 mm to ≈10 µm.

## 3. Results (`python lltc/lltc_hover.py`, seed 0, N_MPC = 25)

All numbers come from `artifacts/lltc_hover/summary.csv` and the companion
CSVs. Timings are wall-clock on the machine that ran the script (IPOPT, one
core) and will differ on yours. The closed-loop cost is
`Σ_t ℓ(x_t, u_t)` over 6 s with the nominal weights.

| quantity | value |
|---|---|
| terminal level `l_T`, `d`, `C_N` | 6.31, 0.0763, 8.14 |
| samples kept by the Ω_N gate | 1500 of 1562 (96 %) |
| fit of V1, R² train / test | **0.9993 / 0.9845** |
| NRMSE test | 0.124 |
| Lyapunov violations (26b) / (26c), all 1500 samples | 3 / 11 |
| closed-loop cost, NMPC / LLTC (mean of 4 states) | 10.17 / 11.17 (+9.9 %) |
| control effort Σ‖u−u_r‖²_Qu, NMPC / LLTC | 2.57 / 3.16 |
| decision variables / equality constraints | NMPC 334 / 234, LLTC 22 / 18 |
| mean solve time T_a, NMPC / LLTC | 8.4 ms / 2.9 ms (**2.9×**) |
| worst-case solve time T_w, NMPC / LLTC | 21.7 ms / 12.7 ms (1.7×) |

### Figures

1. **`fig1_fit_quality.png`**: learned vs true cost-to-go, with R² and the
   relative error against the distance from hover.
2. **`fig2_altitude.png`**: altitude from 4 initial states (±16 cm, ±0.5 m/s,
   ±7°). LLTC is close to NMPC. Both converge to the reference with sub-mm
   error.
3. **`fig3_control_effort.png`**: thrust, body-rate magnitude and total
   effort. LLTC uses ~20 % more effort for a ~10 % higher cost.
4. **`fig4_ood.png`**: initial states at 1 to 4× the training box. LLTC stays
   stable everywhere tested. Its cost gap grows from +16 % (1×) to +28 % (4×),
   because the learned P̂ is extrapolated outside the data.
5. **`fig5_weight_sensitivity.png`**: LLTC only.
   - *Lyapunov penalty λ1 = λ2.* At λ = 0 (no Lyapunov penalty, i.e. plain
     LTC) the fit is the best (R² 0.989) but 1242 samples violate the decrease
     condition and **the closed loop diverges** (cost 4.8·10⁴). Already at
     λ = 0.1 the controller is stable. At λ = 1000 the penalty dominates the
     fit (R² 0.81) and the cost rises. The range 0.1 ≤ λ ≤ 100 is flat.
   - *Stage weights changed online with the learned P̂ kept fixed.* Scaling Qx
     has **exactly no effect**: with N = 1 the state cost only weights the
     fixed x0, so all state weighting lives in the learned terminal cost.
     Scaling Qu by 0.25 or 4 costs +45 % / +29 %, because P̂ was learned for
     the nominal Qu. A change of weights therefore requires re-learning P̂.
6. **`fig6_computation.png`**: average / worst-case solve time, per-step
   histogram, and NMPC solve time against horizon. The speed-up is smaller
   than the paper's ≈9× because IPOPT's fixed per-call overhead (≈2 ms, the
   N = 1 point) dominates a 22-variable problem. A code-generated solver
   (acados) would widen the gap.

## 4. Outputs

`artifacts/lltc_hover/`: `fig1`…`fig6` (`.png`), plus `fit.csv`,
`effort.csv`, `ood.csv`, `weight_sensitivity.csv`, `timing.csv`,
`timing_vs_horizon.csv` and `summary.csv`.

## 5. Relation to `study/notebooks/nb2_lltc.py`

`nb2_lltc.py` is the LLTC inside the study's JAX/iLQR pipeline. It tracks
time-varying references, and its `model.pkl` is consumed by Notebooks 6 and 7.
`lltc_hover.py` is an independent, from-scratch version that follows the paper
closely (CasADi/IPOPT, the Ω_N gate, both Lyapunov penalties) on the simpler
hover task. The two share no code.
