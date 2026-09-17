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

---

## F. Found while correcting, not in the original audit

The three below were introduced or exposed by the corrections themselves. They
are recorded here because two of them changed a reported result.

### F1 — the C5 smoothing misreported the flagship latency by 10×

The rolling mean added for C5 held its predictions in a growing Python list and
called `jnp.stack` on it each step. Until the buffer saturates, every step
produces a **new shape**, and eager JAX compiles per shape.

| buffer state | cost per step |
|---|---|
| filling (steps 1…32) | **68 ms** |
| saturated | **0.29 ms** |

`solve_latency_ms` probes `T = 30` steps from a fresh episode, so the whole
probe landed inside the warm-up. The ledger therefore reported

```
Adaptive AC-MPC N=1   44.9 ms median   112.1 ms p95   admissible=False
```

against a true

```
Adaptive AC-MPC N=1    3.94 ms median    4.23 ms p95   admissible=True
```

The flagship contribution was being called undeployable by a factor of ten, on
the one number that decides deployability, because of how its own smoothing
warmed up. Corrected to a preallocated `(n_s, B, 6)` ring buffer with the mean
taken inside a single jitted call.

The change is **performance-only**: the ring mean is bit-identical to the
growing list across the fill and after saturation (max difference exactly `0`
over 80 steps). Every RMSE in the ledger is unchanged to the last digit, which
is the practical confirmation. Two tests pin both properties.

*Method note.* Two earlier hypotheses were measured and rejected first — the
RDP forward pass is already jitted, and naive smoothing at a fixed shape costs
0.29 ms, not 45 ms. The first microbenchmark missed the real cause precisely
because it pre-filled the buffer, which is the one condition under which the
bug does not occur.

### F2 — the observation normaliser had no variance floor

`normalise_obs` divided by `sqrt(var + 1e-8)`. Several channels are
near-constant within a batch — a frozen preview under position hold, an
integral that has not moved — so `1/sqrt(var)` amplified the first sample that
did move by ~1e4 and took the policy to NaN within three iterations. Floored at
`OBS_VAR_FLOOR = 1e-4` and clipped at ±10.

Related: the model-free MLP arm was left unnormalised while the cost map was
normalised, so §8.3's exploration comparison was measuring the normalisation
rather than the architecture. Both arms now share it.

### F3 — `DMOD_MODE` was used ~90 lines before it was defined

Notebook 5 died with `NameError` after the estimators had already trained.
Hoisted to the config block. The measurement it guards was unaffected and is
worth recording: against the true residual (4.9), the (4.13) moment conversion
is **scenario-dependent, not a fixed factor**.

| scenario | `om_true` | `first_order` | ratio |
|---|---|---|---|
| asym 0.01 | 0.0016 | 0.149 | **95.2×** |
| asym 0.07 | 0.7604 | 1.129 | 1.49× |
| central 0.15 | 0 | 0 | — (no moment) |

On a small standing moment the rate integrator has absorbed nearly all of it
and (4.13) overstates by ~95×; on a large one, by ~1.5×. `closed_loop` at
`DMOD_SETTLE_S = 0.5 s` scales by `moment_gain = [0.901, 0.901, 0.917]` — the
right direction, far too small for the quasi-static case. This is why all three
modes are tabled rather than one being declared correct.

## What the corrected smoke-scale ledger says

`AC-MPC N=1` S1 RMSE moved **0.137 → 0.831**, from better than LQR to worse.
That is the correction working, not a regression. Notebook 3's own header
predicts "every learned controller diverging" at this scale, and the *old* run
contradicted its own documented expectation because exploration noise was not
reaching the action (B1): the policy was effectively the MPC with a near-hand
cost map, which tracks well precisely because it is not learning. With the
noise on the action and a six-iteration smoke budget, training genuinely fails.

**These rows require `X500_SCALE=full` before they can be read as results at
all.** The same applies to the estimator R² (LSTM 0.277, GRU 0.261, CNN 0.160,
TCN 0.030): all four are latency-admissible, and the §9.5 rule correctly selects
on R² among them, but 0.28 is a weak predictor and is a smoke-budget artefact.

## Verification of the corrected tree

Run against the final tree, on an otherwise idle machine:

| check | result |
|---|---|
| `pytest study/tests/` | **83 passed** (23 of them pin audit findings) |
| `study/check_consistency.py` | **all consistency checks passed** |
| `acmpc_controller.check_glue` | **PASSED** |
| notebooks 1–7 | all seven run to completion in one chain |
| `export_estimator.py` ×5 | parity ~1e-14 against a 1e-5 tolerance |

The exported predictors all sit inside the 20 ms control period on the
deployment NumPy path — CNN 0.78, TCN 1.95, GRU 5.24, LSTM/selected 5.94 ms
p95. Note that this is a different measurement from the ledger's `ms_p95`
column, which times the JAX study path via `solve_latency_ms`; F1 above is what
happens when that distinction is not held carefully.

---

# G. The medium-run debug — why no learned controller converged

A third pass, prompted by the `X500_SCALE=medium` run. The symptom that started
it is a contradiction inside that run's own output: Notebook 6 reports
`AC-MPC N=1` at 1.07 m RMSE on **every** suite, **every** path and — in
Notebook 5 — every disturbance scenario and level, a spread of 0.01 m across
conditions that move LQR by 0.28 → 0.95 m; while Notebook 7 films the *same
checkpoints* at 45–50 m, flying 0.09 rad of a 6.28 rad lap. Both numbers are
correct. They measure different things.

`study/tests/test_g_training_regressions.py` pins all of it.

## G1 — the PPO trust region never let a single step land

`train_ppo`'s trust region applies a step, measures the KL, and reverts if the
region was left (B1). The mechanism is right. The bookkeeping around it was not:

1. a trip set `stop = True`, which broke out of the minibatch loop **and** the
   epoch loop, so an iteration attempted at most one update;
2. that one update was reverted — and the revert restored the whole parameter
   tree, so the **critic** and `log_sigma` were rolled back with the actor;
3. `lr` was halved on every trip and grown only under
   `if kl_seen and kl_seen < 0.5 * kl_target`. `kl_seen` is still the float
   `0.0` when the *first* minibatch trips, and Python reads that as falsy, so
   the growth branch was unreachable and the decay was a one-way ratchet.

Measured, on the shipped code at `n_env=32, epochs=4, minib=8`:

| | |
|---|---|
| `optax.apply_updates` calls, 8 iterations | **8** (256 intended) |
| of which reverted | **8** |
| net policy steps landed | **0** |
| `lr` over those 8 iterations | 1.5e-4 → 4.7e-6, halving every iteration, never growing |
| `sigma`, `entropy` | **byte-identical down every column** |

`sigma` is a trained parameter with a non-zero gradient, so a frozen `sigma`
column is proof that nothing was applied. The same signature is in every
training log committed under `artifacts/` — `log_nominal.csv`,
`log_DR-all.csv`, `log_DR-all+noise.csv` all show `lr` halving monotonically and
`entropy` constant at `-1.9127258067248345`.

From `lr = 3e-4`, twelve halvings reach `lr_min = 1e-7`. **`X500_SCALE=full`
does not fix this**: 400 iterations ratchet the learning rate to the floor
exactly as 80 do. The AC-MPC and Adaptive AC-MPC rows of every scale table are
the *initialiser*, which is also why Notebook 3's representation sweep ranks
`chol` (which starts at `S ≈ Q_LO·I`, i.e. with the quadratic term absent, so
the solve is driven by the terminal matrix) above `diag` (which starts at the
sigmoid mid-point, a uniform 3.16 on all 13 channels): that sweep compares three
*initialisations*, not three learned representations.

**Fixed** by making the region a backtracking line search *inside* the
minibatch: halve the step for that minibatch until it satisfies the bound (up to
`kl_backtracks`), keep the critic step when the actor step is rejected, gate the
`lr` growth on whether a step landed rather than on a float being truthy, and
carry on to the next minibatch instead of abandoning the iteration. `train_ppo`
now reports a `landed` column and prints a **NO-UPDATE GATE** banner when it
sums to zero.

## G2 — the reward paid the policy to fly out of the ball

`_step_jit` terminates a vehicle on `bad | far | spin` but charged the −5 crash
penalty only for `bad | spin`. Leaving the 3 m position ball was a **free**
termination, and every term of (5.4) is strictly negative, so cutting the value
bootstrap at that termination makes the return of a terminated episode `r_k`
instead of `r_k + γV`.

Measured with the shipped `gae`, a consistent critic, `r = −3`/step, `γ = 0.99`:

| | advantage at the escape step |
|---|---|
| no termination | +0.00 |
| termination | **+297.00** |
| crash penalty available to oppose it | −5.0 (1.7 % of it) |

PPO was being trained to diverge. That is precisely what Notebook 7 films, and
it is why the TRPO arm — which *does* update (it never had the G1 revert) —
still lands at 1.07 m: it is optimising the reward it was given.

**Fixed** two ways, both needed. `far` now reaches the reward like any other
failure, and `EnvCfg.term` defaults to `'bootstrap'`: the respawn is treated as
a **truncation**, so the value target keeps its `γV(s')` term across the
boundary. The episode boundary is an artefact of the batched environment, not
part of the task. `term='cut'` reproduces the old behaviour.

## G3 — the running observation normaliser was being optimised

`obs_norm` lives inside `params['actor']` so that it travels with the
checkpoint, and `normalise_obs` reads it differentiably. Nothing excluded it
from the optimised tree, so Adam updated `mu` and `var` as if they were weights
— and `obs_norm_update` then folded the corrupted values into the next Welford
step. It also consumed part of the `clip_by_global_norm` budget. `trpo_step` was
worse: `_flat(actor)` put the statistics straight into the natural-gradient
direction, so the TRPO step **overwrote the normaliser**.

**Fixed**: the gradient of `obs_norm` is zeroed before the optimiser sees it
(keeping the pytree shape, so `opt_state` is untouched), and `trpo_step`
optimises over the weight subtree only.

## G4 — the training disturbance was white noise, not a per-episode wrench

`AdaptEnv.step` called `_sample_wrench_dr()` whenever **any** vehicle respawned,
and that method redrew the wrench for the **whole batch**. At `n_env = 128` with
a ~1.5 %/step reset rate, at least one vehicle resets on ~85 % of steps, so the
"constant per-episode wrench" the RDP is asked to infer from a 64-step window
was redrawn at close to the 50 Hz control rate. No window can estimate that.

The window buffer had the mirror-image problem: `_buf_fill` was a single scalar
for the batch and nothing cleared a respawned vehicle's history, so deployment
windows spanned episode boundaries — while `make_windows` refuses exactly that
for the *training* windows (§9.6). Train and test disagreed about what a window
is.

**Fixed**: `_sample_wrench_dr(mask=...)` redraws only the vehicles that
respawned; `_buf_fill` is per-vehicle; `_clear_windows(mask)` drops a respawned
vehicle's history and `ready()` returns a per-vehicle mask.

## G5 — the estimator ran twice per control step for variant C

A variant that routes the estimate to **both** the observation and the model
(C) reaches `d_channel()` twice in one control step: once through
`ctrl_from_actor`'s `dmod_fn` and once through `Env.obs()`. Each call pushed a
fresh entry into the 32-frame rolling mean, so C's smoothing window was half the
paper's length while A's and B's were full length — an unintended difference
between the arms being compared. **Fixed** by caching the prediction per buffer
version.

## G6 — the model-correction variants were never trained with the correction

`train_ppo` reads the residual into the prediction dynamics only when
`cfg['use_d']` is set. Notebook 5's `CFG` never set it. Variants **B** and **C**
— the two whose entire definition is "the residual goes into the model" — were
therefore trained on nominal prediction dynamics and evaluated with `dmod_fn`
feeding the residual into `fc()`. The same mismatch applied to the **Oracle**
arm, which was evaluated with `to_model=True` although it trained without it;
Oracle is the reference the whole "value of information / cost of estimation"
decomposition is measured against.

**Fixed**: variants train with `use_d` matching their `to_model` flag and with
the estimator attached to the *training* environment, so the policy learns
against the prediction it will be deployed with; Oracle evaluates with
`to_model=False`, its privilege being the observation channel.

## G7 — the LLTC fit scored a different object from the controller

Two independent problems in Notebook 2, both consistent with `R² = 0.9999`
sitting next to an LLTC row 7.5× worse than plain NMPC N=1 at 26 % saturation:

- **wrong argument.** `V1 = J − ℓ₀` is the cost-to-go from `e₁`, and
  `make_lltc_ctrl` scores `½ e₁ᵀ P(e₀) e₁` as the terminal cost of the N=1
  problem. The fit regressed `½ e₀ᵀ P(e₀) e₀` onto `V1`. The form that was
  validated is not the form that is used.
- **the acceptance gate was a tautology.** `keep = V1 > 0 & V1 < percentile(V1,
  95)` accepts 95 % of any input, which is why `acceptance` reads `0.9492` in
  *every* row of the terminal-weight sensitivity table, unchanged while `reach_m`
  moves 1.07 → 0.29 m. It was printed as a diagnostic beside the quantity it
  does not respond to.

A third, not a bug but worth stating: `P_θ(e) = L(e)L(e)ᵀ + εI` with `L` an MLP
of `e` makes `½eᵀP_θ(e)e` an arbitrary non-negative function of `e`, so a high
R² is a statement about MLP capacity on 243 points, not about a *quadratic*
terminal cost. And candidates were drawn on one shell at `‖e‖ ≈ 0.8 m` while the
closed loop lives at `‖e‖ ≈ 0.05 m`, so the controller queries the network a
factor ~15 outside its fit support.

**Fixed**: the fit contracts with `e₁`, the gate is `finite ∧ V1 > 0 ∧ ‖e₁‖ ≤
2·reach`, and candidates are sampled over decades of magnitude rather than one
shell.

## G8 — the PID tilt limit was dead code

`make_pid_ctrl` clamped `zb_des` to `tilt_max` and then never read it: `q_des`
came from the unclamped `a_des` and the collective from the unclamped norm. The
collective also used the bare square root, ignoring the idle floor that (4.8),
`hover_u` and `uref_from_traj` all invert (A2) — 5.6 % high at hover. PID is
excluded from the study comparison (§5.12) and is only the ROS incumbent of
§9.8, so nothing in the tables moves. **Fixed** anyway.

## G9 — RMSE under respawn is a property of the bound

The `Env` docstring already names this ("The 3 m bound is a measurement trap,
§5.10"), and `study_moderate.py` opens with the symptom. It is worth converting
into a number, because the medium-run tables were still read as tracking errors.

A cost map that was **never trained** — a fresh `costmap_init`, which after G1
is what every AC-MPC checkpoint contains — measures:

| respawn | T | rmse | maxerr | sat |
|---|---|---|---|---|
| on | 150 | 0.892 | 2.898 | 0.286 |
| on | 300 | 0.904 | 3.135 | 0.332 |
| **off** | 150 | **11.39** | 36.60 | 0.177 |
| **off** | 300 | **41.36** | 117.89 | 0.244 |

(`hid=256`, `ilqr=10`, seed 0. Whether a *particular* random draw diverges is a
property of that draw — a smaller net sometimes holds — but the medium run's own
checkpoints plainly did, which is what Notebook 7 films.)

With the bound on, RMSE does not grow with the rollout length, because `|e_p|`
is truncated at 3 m; without it, it grows without limit. 0.89–0.90 m against the
medium run's 1.07 m, and 41 m against Notebook 7's 45–50 m. Notebook 7 is the
only place `no_respawn` is set, which is the entire explanation for the
Notebook 6 / Notebook 7 contradiction.

**Fixed** by making it visible rather than by changing it: `stats()` now returns
`bound_frac` and `bounded`, and Notebook 6 names the cells where `bound_frac`
exceeds 0.5 % and says their `rmse` is not comparable with the rows that never
respawned.

## G10 — MPVE prices its predicted trajectory with a different reward

Not a bug, but it was undocumented and it biases the critic. `mpve_value_loss`
builds `r_hat` with `om` and `d_u` passed as **zeros**, because the 10-state
control model has no body-rate state and predicts no command increment
(`e[6:9]` is the tilt error `delta`, not `omega`). So the predicted reward drops
the `-0.02|om|²` and `-0.05|Δu|²` terms of (5.4) and is systematically less
negative than the reward the ordinary TD target in `l_v` is built from.
Equation (9)'s consistency term is therefore regressing `V` onto targets from a
different reward than the one it is being fitted to, optimistic in proportion to
how hard the policy is working the rates. Documented in the docstring; no code
change, because the quantities genuinely are not available in the prediction.

## Does `X500_SCALE=full` fix any of this?

No. G1 makes the iteration budget irrelevant — the learning rate reaches its
floor in twelve iterations at any scale — and G2 means that once G1 is fixed,
more iterations buy a *better* escape policy, not a better tracker. G9 means the
tables cannot show either outcome. The undertrained-policy banner that fires in
Notebooks 3 and 6 correctly identifies that the split falls along the training
axis; it attributes it to the scale table, and the scale table is not the cause.
