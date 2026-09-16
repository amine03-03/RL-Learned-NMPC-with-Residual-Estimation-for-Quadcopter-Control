# Audit against the source papers and the real PX4 x500

A second pass over the whole repository, checking the implementation against
three external authorities rather than against the build specification:

| authority | used for |
|---|---|
| [PX4-gazebo-models @ `bb0b9cf`](https://github.com/PX4/PX4-gazebo-models/tree/bb0b9cf974acf4f1bcb5f5fcf80b88841562dea9/models) — `x500`, `x500_base` | every vehicle constant |
| the `gz_x500` PX4 airframe file (`SIM_GZ_EC_*`, `CA_ROTOR*`, `MPC_THR_HOVER`) | the actuator and allocator interface |
| Romero, Aljalbout, Song & Scaramuzza, *Actor-Critic MPC*, IEEE T-RO 2025 (**arXiv:2306.09852**) | the AC-MPC learning architecture |
| Saj, Vemuri, Kalathil & Benedict, *Adaptive Outer-Loop Control of Quadrotors via RL*, VFS 82 (**arXiv:2605.16015**) | the adaptive mechanism and the RDP |

This project substitutes **AC-MPC for the adaptive paper's RL policy**; the
adaptive *mechanism* (oracle → RDP, the 26-D causal frame, the payload and slung
scenarios) is the paper's, the *controller* is not.

Everything below was found by measurement. `study/tests/test_audit_corrections.py`
pins all of it (19 tests).

---

## What was already right

- Every constant the SDF fixes — `m_body`, `m_rotor`, `J_body`, `K_T`, `k_m`,
  `Ω_max`, `τ_up/τ_dn`, all four rotor positions, the ccw/ccw/cw/cw spin
  pattern — matches **exactly**, and the PX4 `CA_ROTOR*` index mapping is
  consistent once FRD→FLU is applied.
- `base_link` has no `<inertial><pose>`, so the airframe CoM really is at the
  model origin, as assumed.
- **The iLQR is exactly optimal**: against an independent Powell optimiser on
  the same objective, the relative gap is **1e-13** at N = 1, 3 and 10.
- Feeding the converted residual into the control model improves one-step
  prediction by **3.3× to 688×**, with every sign correct.
- Rewards (5.4)/(5.5), GAE, the error coordinates and the export parity all
  reproduce their closed forms exactly.

## A. Physics against the real platform

### A2 — the PX4 idle floor was ignored *(largest single error)*

`SIM_GZ_EC_MIN = 150`, so PX4 maps the normalised command onto
`Ω = 150 + 850·c`, **not** `c·Ω_max`. Thrust is still quadratic in rotor speed,
but it is not proportional to `c²`.

| | before | corrected | error |
|---|---|---|---|
| `u_hover` | 0.769431 | **0.728742** | 5.58 % |
| `∂a_z/∂c` at hover | 25.4905 | **21.6670** | **17.65 %** |
| thrust at `c = 0` | 0 N | **0.7694 N** (3.8 % of weight) | the floor did not exist |

The slope error propagated into `K_LQR`, the Riccati terminal matrix and every
feed-forward collective. `thrust_of`, `hover_u`, (4.8) and `uref_from_traj` now
invert the affine-in-Ω map, and `omega_of_cmd`/`cmd_of_omega` are the one place
it is defined.

### A3 — the allocator and the rotors disagree about `k_m`, and that is real

The Gazebo plugin produces `momentConstant = 0.016`; PX4's allocator believes
`CA_ROTORn_KM = 0.05`. A commanded yaw torque therefore realises **32 %** of its
intent on the real stack. The repo previously used 0.016 for *both*, so the
mismatch existed nowhere. `P.k_m_ctrl` is now carried separately and
`par['Minv_ctrl']` is built from it, so `cond(M_ctrl) = 20` against the plant's
62.5 and the yaw error is in the model where it belongs.

### A4 — PWM normalisation did not match `actuator_motors`

`inner_loop` returned `Ω_cmd/Ω_max`; PX4 publishes the command normalised on
`[Ω_min, Ω_max]`. Different slope *and* offset on 4 of the RDP's 26 input
channels — a train/deploy gap on the block §6.2 calls the key design decision.
Now `cmd_of_omega(Ω_cmd)`.

### A5, A6 — SDF terms that were missing

`rotorDragCoefficient = 8.06428e-05` and `rollingMomentCoefficient = 1e-06` are
now modelled (both act on the airspeed component perpendicular to the rotor
axis and scale with rotor speed), and the rotors carry their **own** inertia
instead of being point masses (+0.006 % / +0.44 % / +0.24 % on `J`).

## D7 — 62 % of the roll/pitch command box was unreachable

`om_max = (10,10,4)` rad/s but the autopilot clips the rate command at
`rate_max = 3.84` rad/s, so `|ū| > 0.384` on roll and pitch asked for a rate the
plant could not produce — and `fc()` had no such clip, so the optimiser planned
freely in a region where model and plant disagreed.

The fix is **not** a clip inside `fc()`: that was tried and leaves the
derivative zero beyond the knee, so `Q_uu` goes singular in the rate directions
exactly when the solver tries to come back, and the collective absorbs the
difference — measured, untrained saturation went to **47 %**. Instead the input
box *is* the reachable set, `U_HI = [1, 0.384, 0.384, 0.960]`, so model and
plant agree everywhere inside it and there is no dead region. Saturation
returned to 0 % for LQR and NMPC.

## D3/D4 — (4.13) is a short-transient model used in a quasi-static regime

(4.13) neglects `K_i`. Measured closed-loop gain of the rate loop to a step
moment — the factor by which the true rate residual differs from (4.13):

| t [s] | 0.1 | 0.3 | 1.0 | 3.0 | 8.0 |
|---|---|---|---|---|---|
| gain | 0.87 | 0.93 | 0.76 | 0.43 | **0.103** |

So for a *standing* moment — exactly the `asym` payload scenario — the converted
residual was **≈9× the true one**, and variants B and C fed that into the
prediction dynamics. `moment_gain(t)` now derives the gain analytically (it
reproduces the measurements to ~0.02), `wrench_to_dmod` gains a `closed_loop`
mode, and Notebook 5 prints the validity table beside the results.

## D5 — the helix was inconsistent when the feasibility cap bound

`path_demand` under-reported peak speed by up to **20 %**, and the climb term
used the *sampled* `spd` while the horizontal speed was `R·ω ≠ spd`. The climb
now follows the realised `R·ω`, which also restores the dagger condition
`KAPPA_V[helix]` is stated under.

## B. AC-MPC against arXiv:2306.09852

### B1 — exploration noise was in the wrong place *(voided an experiment)*

The paper: `u ~ N{diffMPC(x, Q(s), p(s)), Σ}` — noise on the **action**. The
repo perturbed the **cost-map parameters**. Measured saturation against σ:

| σ | parameter noise (before) | action noise (paper, now) |
|---|---|---|
| 0.05 | 6.2 % | 10.9 % |
| 0.15 | 6.2 % | 15.6 % |
| 0.30 | 4.7 % | **29.7 %** |

§8.3's exploration study is *about* noise driving a box-constrained collective
onto its box. Under parameter noise that mechanism cannot occur, so F10's
saturation axis was measuring something unrelated.

Consequence, and it is the paper's too: the MPC is now **re-solved on every
minibatch of every epoch**, because the policy mean depends on θ through the
solve. Training is correspondingly slower.

This exposed a real property of the architecture: the MPC mean is far more
sensitive to its cost map than an MLP's output is to its weights — one clipped
Adam step moves μ by 0.32 for a 0.5 % change in `S`, which at σ = 0.05 is
already KL 4.8. PPO therefore needed a genuine trust region: the step is
applied, the KL measured, and the step **reverted** if it left the region, with
the learning rate adapted on the same signal. TRPO, which enforces this by
construction, holds KL at its target exactly and is the better-behaved arm here.

### B2 — the linear cost term `p` was never learned

The paper's cost-map output is `2T(n+m)`: `Q` **and** `p`. `costmap_from_z`
returned `c ≡ 0` unconditionally and `with_c` defaulted off, so `p` was
identically zero everywhere. `REP_DIM['diag']` is now 26 per stage (13 + 13) and
`p = P_HI·tanh(z)`. The bound is **symmetric**, unlike the paper's positive-only
sigmoid: in *error* coordinates a strictly positive `p` biases every channel one
way. `P_HI = 2.0` is kept below the geometric-mean quadratic weight 3.16, since
the unconstrained minimiser of `½St² + pt` sits at `t = −p/S`.

### B3 — MPVE was a dead config flag

`mpve_targets` was defined and never called; `cfg['mpve']` appeared only in the
defaults. Notebook 3's entire MPVE sweep produced **bit-identical rows** and a
`gain` column of exact zeros, under prose discussing "expect the largest gain at
low λ". Equations (8) and (9) — including the TD-k consistency term — are now
implemented in `mpve_value_loss` and added to the critic loss.

One documented approximation: the critic is `V(o)` over the 40-D observation
while the MPC predicts the 9-D error, so a predicted observation substitutes the
predicted error and holds the preview/integral blocks. Over 20–200 ms those move
very little, but it is an approximation, not an identity.

### B4, B5 — normalisation and exploration

Observation normalisation (running mean/std, refreshed once per iteration and
travelling with the checkpoint) now exists, per §IV-A of the paper. It must be
refreshed at the **end** of an iteration: updating it between the rollout and
the update makes `lp_old` and `lp` come from different policies — measured, KL
jumped to 2.1e4 on the first minibatch and the parameters went NaN one iteration
later. `log_sigma` is now a learned parameter rather than a frozen constant.

## C. The adaptive mechanism against arXiv:2605.16015

| | before | corrected |
|---|---|---|
| RDP window `H` | 32 | **64** (paper) |
| GRU widths | (128, 64) | **(64, 64)** (paper) |
| deployment smoothing | first-order IIR, default off | **rolling mean of 32** (paper) |
| `asym` brackets | 0, **4**, 7, 11 % | 0, **1**, 7, 11 % (paper) |
| slung tether | `L = 0.5 m` = 2.87 arms | **`L = 0.174 m` = 1 arm** (paper) |
| DR moment fraction | 0.098 of weight·arm | **0.1156** (paper, non-dimensionalised) |
| training objective | circle/fig8 **tracking** | **position hold** (paper) |

The tether change matters physically: the pendulum period goes from 1.42 s to
**0.837 s**, a different disturbance bandwidth relative to the airframe.

### Position-hold training

The paper trains the policy *and* the RDP on a position-hold objective and
argues this is what "promotes stable, aggressive recovery from severe
disturbances while naturally generalizing to dynamic trajectory tracking".
Training on a moving reference instead mixes tracking error into the very signal
the RDP must isolate.

`EnvCfg.task` — previously a field nothing read (E3) — now selects it. A hold
reference is a **genuine fixed point**: `v_ref = 0`, `a_ref = 0`, hence
`q_ref = identity` and `u_ref = hover`. Freezing only the clock would leave
`v_ref` at its t = 0 value (1.5 m/s on a circle) while telling the vehicle to
hold, which was the first attempt and was wrong.

Notebook 5 trains Base/Robust/Oracle, all three variants and the RDP on hold,
and **evaluates on the tracking suites**, so the generalisation claim is tested
rather than assumed. In the workspace, `reference_generator` gains a `hold`
mode and the experiment plan gains a phase **E-0** (15 runs) that generates the
RDP training set on hold before E-A evaluates on (9.5).

## Secondary

- **E1** — the oracle observation block is a wrench in N/N·m, 2.3× the rms of
  the rest of the vector; resolved by B4's normalisation.
- **E3** — `EnvCfg.task` was dead; now it selects the training objective.
- **E4** — `oracle_target='residual'` was never exercised; the residual path is
  now reachable and the (4.13) validity table is printed beside it.
- **The cost map existed in two copies** that had to agree with nothing checking
  they did; they are one function, `costmap_from_z`.
- The hand-tuned operating point was **re-derived by measurement** under the
  corrected physics: `Q_pos = 10`, `R = 5`, RMSE 0.0506 m, best-to-worst ratio
  **20.8×** over the admissible grid.

## Not corrected, and why

- **`MPC_THR_HOVER = 0.6`** in the airframe file is PX4's own hover-throttle
  estimate and sits in a different normalisation from the actuator command; the
  derived value under the corrected map is 0.729. Left alone, and noted here.
- **`EKF2_MCOEF = 0.12`** is an estimator momentum-drag parameter, not a plant
  one. It does not enter the dynamics this repository simulates.
- The bulk drag `D = (0.30, 0.30, 0.35)` remains a **declared modelling
  addition** on top of the SDF's rotor drag, as §2.1 states.
