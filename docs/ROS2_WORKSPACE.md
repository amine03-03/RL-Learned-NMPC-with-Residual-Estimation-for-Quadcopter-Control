# `rdp_acmpc_ws` — what is in the ROS 2 workspace and why

This document explains the deployment half of the project: the PX4 / Gazebo
workspace that implements §9 of `AGENTS_SPEC_Adaptive_ACMPC.md`. The study half
(`study/`) trains the controller and the predictor; this workspace flies them.

**The claim the workspace exists to test**, stated so it can fail:

> A lightweight causal RDP can estimate residual plant dynamics online and
> supply useful disturbance information to ACMPC **without violating the
> real-time control budget.**

Three levels must *all* hold: **prediction** (quantified estimation error),
**control utility** (closed-loop improvement over nominal ACMPC on *unseen*
disturbances), and **real-time feasibility** (quantified timing against a 20 ms
period). Nothing here claims physical generalisation beyond the tested
Gazebo/PX4 model.

---

## 1. The architecture decision, and why

Every package is split into a **pure-Python algorithmic core** and a **thin
`rclpy` node wrapper**. The core imports with no ROS on the path; the wrapper
prints a clear message and exits 1 if `rclpy` or `px4_msgs` is missing.

Three reasons, in order of importance:

1. **The logic is testable without a graph.** Frame conversions, the `B_d`
   bridge, the scenarios, the ring buffer, the watchdog and every metric are
   exercised by `study/tests/` and by the two check binaries. A control-law bug
   and a frame bug are different bugs, and §11 insists they be caught by
   different tests.
2. **The RDP must not depend on JAX or PyTorch.** The estimator runs inside a
   20 ms budget and cannot afford a framework. Its forward pass is
   `study/rdp_infer.py` — pure NumPy — and the weights arrive as a
   framework-neutral `.npz`.
3. **The controller deliberately *does* import the study module.** It would have
   been easy to reimplement the dynamics here in NumPy. That would be a second
   model to keep in step with the first, and no test could see them diverge.
   Instead `acmpc_controller.controller` imports `x500_core_jax`, and
   `check_glue` asserts there is exactly one set of constants.

The asymmetry is the point: **one model, one set of constants, and a
framework-free estimator.**

## 2. Data flow, and how causality is enforced

```
reference_generator ──/reference/trajectory──┐
                                             v
  PX4 ──/fmu/out/vehicle_odometry──►  acmpc_controller  ──/fmu/in/vehicle_rates_setpoint──► PX4
       ──/fmu/out/actuator_motors──►         ▲                /fmu/in/offboard_control_mode
                     │                       │
                     v                       │
              rdp_estimator ──/rdp/disturbance_estimate──┘
                     │
disturbance_manager ─┴──/disturbance/ground_truth──► state_logger, visualization  (ONLY)
```

**Ground truth never reaches the online controller.** §9.1 requires this, and it
is enforced structurally rather than by convention: the RDP's window is built by
`rdp_estimator.ring_buffer.RingBuffer`, which is fed only from state and actuator
topics. The disturbance manager has no handle on it, so there is no call by which
an injected disturbance value could enter a window. `/disturbance/ground_truth`
is subscribed by the logger and the evaluator and by nothing else.

The same trap exists one level up, inside the study, and it bit this
implementation: the oracle observation channel was returning the *truth* even
with an RDP attached, which made every "RDP" row secretly an oracle row. That is
correction **N-8** in `CORRECTIONS.md`, and a regression test now pins it.

## 3. The packages

### `acmpc_controller` — the control node, frames, `B_d`, and the preflight

| file | what it is |
|---|---|
| `frames.py` | PX4 ↔ study frame conversions and the CTBR exit mapping |
| `bd_bridge.py` | §4.4's wrench → model-residual conversion, **implemented once** |
| `controller.py` | the ROS-free control core; modes `acmpc`, `nmpc1`, `pid` |
| `controller_node.py` | the 50 Hz `rclpy` wrapper |
| `metrics.py` | §9.12: RDP accuracy, adaptation time, closed loop, timing |
| `check_ctbr.py` | the four preflight checks of §9.4 |
| `check_glue.py` | workspace ↔ study constants agreement |

**Frames.** Everything inside the package is ENU/FLU; PX4 is NED/FRD on the
wire, and the conversion happens at the node boundary and nowhere else.
`q_ENU/FLU = q_NED→ENU ⊗ q_PX4 ⊗ q_FRD→FLU`, rates exit as
`roll = ω_x, pitch = −ω_y, yaw = −ω_z`, and `thrust_body` is FRD down-positive,
so a positive collective is `[0, 0, −c]`.

`QuaternionDifferentiator` is the `ω̇` source of last resort: it uses
`timestamp_sample` and **not** the wall clock, keeps the quaternion hemisphere
continuous, floors `dt` at 2 ms, rejects `|ω| > 12 rad/s` and low-passes at
15 Hz. The rejection count is exposed and shown on the live panel — a drained
queue can otherwise produce `|ω| = 15` against a true 0.4, and the plot looks
fine.

**`bd_bridge`** is the one place the workspace converts a wrench into something
the prediction model can use: force → acceleration via `1/m`, moment → commanded
rate via (4.13). Getting it wrong is silent — the raw wrench is a factor
`m = 2.064` too large in force and dimensionally unrelated in moment, and it
raises no exception, it just produces plausible numbers that are wrong in the
same direction every time. `check_glue` compares it against the study's
`wrench_to_dmod` on 64 random wrenches.

**The three controller modes** exist because §9.8 wants the deployment result to
connect to the study's ledger: `acmpc` is the proposed method, and `nmpc1` and
`pid` are the realistic incumbents, run on S0/S1.

### `rdp_estimator` — ring buffer, NumPy inference, watchdog, timing

`RingBuffer` holds `H = 64` frames (1.28 s at 50 Hz, the paper's window) of the 26-D frame of (6.2):

```
[ p − p_ref (3) | vec(R) row-major (9) | v (3) | ω (3) || u_{t−1} − u_ref (4) || PWM (4) ]
```

`ready()` is False until `H` real frames have arrived; until then the estimate is
forced to zero rather than being read off zero padding.

**The PWM block is the key design decision of the whole adaptive half**, and this
implementation verified the justification rather than assuming it. Applying a
constant `τ_x = 0.20 N·m` to the rate loop alone:

| t [s] | ‖ω‖ [rad/s] | PWM spread | I_ω,x |
|---|---|---|---|
| 5 | 1.46e-1 | 0.0437 | −1.588 |
| 20 | 2.00e-3 | 0.0437 | −2.091 |
| 80 | **7.15e-11** | **0.0437** | −2.098 |

The rate error goes to zero; the mixer spread holds forever; `I_ω` converges to
the predicted `τ/(J·K_i) = 2.098`. *State and setpoint alone cannot resolve a
standing moment; the actuator commands can.* This is why §9.4's fourth check
refuses to start a moment-producing scenario when `actuator_motors` is not
publishing — without it the moment channels are unobservable and the dataset is
worthless.

`Watchdog` runs the forward pass under a soft budget and, on failure or a bad
output shape, **falls back to `d̂ = 0` and counts a fault**. A fallback is a
result, not something to swallow: a run with a nonzero fault count has to say so
in its metrics. `TimingRecorder` produces mean / p95 / p99 / max and the
deadline-miss rate per component. `CausalFilter` is the optional first-order
filter of §9.11 step 4 — first order and causal by construction, so applying it
cannot violate §9.1.

### `disturbance_manager` — S0…S6

Deterministic and reproducible, seeds recorded per run. `Scenario` returns the
true external wrench and, separately, the plant-parameter perturbation the ACMPC
is **not** told about (it keeps nominal parameters — that mismatch is the thing
to be estimated).

| id | scenario | purpose |
|---|---|---|
| S0 | nominal | baseline stability and tracking |
| S1 | parameter mismatch: mass +10…20 %, inertia ±5…15 %, CG offset 2–5 cm | residual-model learning |
| S2 | bounded force/moment step over a finite interval | adaptation time `T_adapt` |
| S3 | finite-duration time-varying gust, smooth at both ends | transient estimation |
| S4 | periodic `F_x = A sin(2πft)`, f ∈ {0.1…4} Hz, plus a chirp | prediction bandwidth |
| S5 | one motor's effectiveness reduced by a known factor | actuator/model residual |
| S6 | combined stress | final demonstration |

Episode timeline: 0–5 s nominal, 5–12 s disturbance active, 12–15 s recovery, so
the onset is visible.

### `reference_generator` — (9.5) with a feasibility gate

`x_d = A cos(ωt)`, `y_d = B sin(2ωt)`, `z_d = z₀`, at three aggressiveness
levels. **The node refuses to publish an infeasible reference.** Measured against
the `α·a_lat_max = 8.0098 m/s²` budget at `A = 1.2, B = 0.9`:

| level | ω [rad/s] | peak v [m/s] | peak a [m/s²] | |
|---|---|---|---|---|
| gentle | 0.60 | 1.298 | 1.332 | feasible |
| moderate | 1.00 | 2.163 | 3.700 | feasible |
| aggressive | 1.45 | 3.137 | 7.779 | feasible, near the edge |

The largest feasible ω at that amplitude is **1.4713**, so "aggressive" sits just
inside it. This matters because a benchmark built on an infeasible reference
measures the reference, not the controller — the same reason §4.2's cap exists in
the study, where the uncapped superellipse demanded 40.85 m/s² against a
13.35 m/s² envelope.

### `state_logger` — 50 Hz synchronised CSV

47 columns per sample: time, episode, phase, state, reference, command, motor
commands, true disturbance, predicted disturbance, the three timings, the RDP
fault flag and its ready flag. The explicit `episode` column is what lets
training split **by complete episode, never by shuffled samples**, and lets
`split_frames` drop windows that would straddle a boundary.

### `experiment_manager` — sweeps and config capture

Plans **118 runs**: E-A (96) the S0…S6 × C0…C3 matrix at three seeds plus the PID
and NMPC incumbents on S0/S1; E-B (7) the frequency response; E-C (12) the MPC
horizon interaction; E-D (3) history length.

`capture_config` records the git commit and dirty flag, PX4 and Gazebo versions,
ROS distro, host, rates, horizon, `H`, seed, plant parameters and the calibrated
`T_max`. **Fields it cannot read say `unavailable`** and the config gains a
`reproducibility_warning` naming them, rather than the field quietly going
missing.

`summarise` reports median and IQR across seeds — never a single seed's number.

### `visualization` — R-F1 live panel and the offline figures

`live_panel` is the 2×3 grid of the six channels, truth solid against prediction
dashed, with the **analytic guide (9.7) drawn as a horizontal line**. That guide
has no free parameters, which makes it the most convincing validation available
and the fastest way to see a sign or scale error. Rolling 2 s R² sits in each
corner and ‖p − p_d‖ in an inset. The title carries scenario, controller,
calibrated `T_max`, RDP p95, `actuator_motors` OK/MISSING and the rejection
count — a panel that looks right while `actuator_motors` is missing is a panel
whose moment channels are unobservable.

Also implemented: R-F2 scatter with the identity line (a sign error shows as a
reflected line), R-F3 steady state against (9.7), R-F4 tracking with onset and
removal marked, R-F5/R-F6 frequency response, R-F8 timing histogram with the
20 ms deadline and p99 marked, and R-F9 the horizon trade-off with infeasible
horizons shaded.

## 4. The two check binaries

§11 is explicit that *a green control-law test says nothing about frames or
constants* — keep them separate. So there are two, and the study has a third
(`study/check_consistency.py`) written independently, so that one edit cannot
silently satisfy both sides.

```bash
ros2 run acmpc_controller check_ctbr    # §9.4, four checks, any failure aborts
ros2 run acmpc_controller check_glue    # constants and the B_d bridge
```

**`check_ctbr`** results as measured here:

| check | result |
|---|---|
| 1. frame round trip, 500 random attitudes | **2.22e-16** (composed vs direct 1.33e-15; tilt heading spread **0.0**) |
| 2. CTBR mapping | **9/9 assertions pass** |
| 3. thrust calibration | SKIP — needs a measured hover collective (`--y-measured`) |
| 4. `actuator_motors` publishing | SKIP — needs a live graph |

**A skip is not a pass**, and the binary says so and tells you what to re-run.
Passing `--moment-scenario` makes check 4 *blocking*, so a moment-producing
recording cannot start without it.

Check 3 is where the subtlest trap lives. Two nonlinearities compose — the
allocator applies `c = (1−f)y + fy²` and the motor gives `F = F_max y²` — and on
this airframe `f = 0`, so `c = y` and `y_hover = √(mg/T_max) = 0.769431`. Using
an uncalibrated `T_max` in the *truth* while the thrust model uses the calibrated
one biases **every recorded `F_ext` in the same direction**: a systematic error
invisible in any per-sample check. That is why `T_max` is derived in exactly one
module and every consumer is asserted against it.

**`check_glue`** passes: `m`, `T_max`, `g`, `J`, `K_rate`, `u_hover`,
`a_lat_max`, `α`, the scenario constants and the arm length all agree with the
study; the two `B_d` bridges agree to **1e-12** on 64 random wrenches; the frame
width and the six channel names and units match; and every exported `.npz` is
loaded and run forward.

## 5. The export bridge

`study/export_estimator.py` is the blocking dependency between the halves. It
loads the JAX pytree, writes a framework-neutral `.npz` with a JSON metadata
blob, and **asserts parity** between the JAX predictor and the pure-NumPy
`rdp_infer` on 64 random windows. **Parity failure fails the export** — a
deployment model that has silently diverged from the trained one is the most
expensive bug available here, because nothing downstream would notice.

All four encoders are exported into `models/`:

| model | parity (NumPy vs JAX) | p95 latency | verdict |
|---|---|---|---|
| `rdp_gru.npz` | 1.33e-15 | — | admissible |
| `rdp_lstm.npz` | 2.32e-15 | 1.44 ms | admissible |
| `rdp_tcn.npz` | 1.08e-14 | — | admissible |
| `rdp_cnn.npz` | 6.22e-15 | — | admissible |
| `rdp_selected.npz` | 2.32e-15 | **1.44 ms** | the shipped one, `H = 64` |

**The encoder is chosen on latency, with R² as the objective inside the
admissible set** — the reverse of the usual ordering, and the one a 20 ms period
demands. A predictor at 137 ms with R² = 0.88 is not deployable; one at 2 ms with
R² = 0.82 is.

## 6. Configuration

`config/acmpc.yaml`, `rdp.yaml`, `disturbances.yaml`, `experiments.yaml`. The
vehicle constants in `acmpc.yaml` are **not a second source of truth** — they are
asserted against the study by `check_glue`, and exist so a deployment can see
what it is flying.

`experiments.yaml` fixes the `T_adapt` convergence criterion (normalised error
below 0.2 held for ≥ 0.5 s) **before** the final evaluation, as §9.12 requires.
Deciding it afterwards is how an adaptation time becomes whatever the author
wanted. Related: `metrics.adaptation_time` bounds its search at the
disturbance-off instant, because without that bound a predictor stuck at zero
"converges" the moment the disturbance stops and scores `T_adapt = 7.0 s` on a
5–12 s disturbance instead of the `NaN` it deserves.

## 7. What is verified here, and what is not

**Verified in this repository, by running it:**

- every module imports and every algorithmic core runs;
- `check_ctbr` checks 1 and 2, and `check_glue` in full;
- the `B_d` bridge against the study's, to 1e-12;
- the RDP data path end to end: frame → ring buffer → NumPy inference →
  watchdog → filter, including fault injection and the zero fallback;
- the S0…S6 definitions, the reference feasibility gate, all the metrics
  (including the adaptation-time edge case above), and every offline figure;
- JAX ↔ NumPy parity for all four exported encoders;
- the 118-run experiment plan and the config capture, including its warning path.

**Not verified here, because it needs a live PX4 SITL + Gazebo graph:**

- `check_ctbr` checks 3 and 4 — thrust calibration needs a measured hover
  collective, and `actuator_motors` needs a publisher;
- §9.15's S0 zero-check (`|F_z| < 0.3 N`, `‖τ‖ < 0.05 N·m` in steady hover) and
  the analytic steady-state check within 10 %;
- the E-A…E-D runs and therefore tables R-T1…R-T5;
- measured timing against the 20 ms deadline *on the target machine* — what is
  measured here is `rdp_infer` p95 at 1.44 ms on this container, recorded in the
  `.npz` metadata, which is the estimator's share of the budget and not the loop.

The node wrappers are written against the PX4 message interfaces and will need
the usual first-run shakedown against a real graph. Nothing in this section is
claimed as flown.

## 8. Building and running

```bash
# the estimator must be exported before the node can load it
cd study && python export_estimator.py --arch GRU --out ../rdp_acmpc_ws/models/rdp_gru.npz

cd ../rdp_acmpc_ws && colcon build && source install/setup.bash

# preflight -- any failure aborts the run
ros2 run acmpc_controller check_ctbr --moment-scenario
ros2 run acmpc_controller check_glue

# the nodes
ros2 run reference_generator  reference_node  --ros-args --params-file config/acmpc.yaml
ros2 run rdp_estimator        estimator_node  --ros-args --params-file config/rdp.yaml
ros2 run acmpc_controller     controller_node --ros-args --params-file config/acmpc.yaml
ros2 run disturbance_manager  manager_node    --ros-args --params-file config/disturbances.yaml
ros2 run state_logger         logger_node     --ros-args -p run_dir:=runs/experiment_0

# offline figures from a finished run
ros2 run visualization live_panel runs/experiment_0
```

Without a ROS environment the cores still run directly:

```bash
PYTHONPATH="rdp_acmpc_ws/src/acmpc_controller:rdp_acmpc_ws/src/rdp_estimator:study" \
  python -m acmpc_controller.check_ctbr
```

## 9. Position-hold RDP training (E-0)

arXiv:2605.16015 trains the adaptive policy **and** the RDP on a position-hold
objective, arguing that is what produces aggressive disturbance recovery which
still generalises to tracking. Training on a moving reference mixes tracking
error into the signal the RDP has to isolate.

The workspace follows that: `reference_generator` has a `hold` mode
(`level: hold`) whose reference is a genuine fixed point — `v_d = 0`, `a_d = 0`,
not a frozen clock on a moving curve — and the experiment plan opens with
**E-0**, 15 runs across S0/S1/S2/S3/S5 at three seeds that generate the RDP
training set on hold. E-A then evaluates on (9.5). `config/rdp.yaml` records
`train_reference: hold` so a run cannot silently be trained on the wrong one.

## 10. Where the specification was wrong

Two of the eight corrections in `CORRECTIONS.md` are about this workspace's
subject matter and are worth repeating here:

- **C-6** — §11's T-12 accepts on `‖τ_ext‖` being 5× larger under the nominal
  mixer. At equilibrium (4.11) reduces to `τ_ext = −m_p g (0.174, −0.174, 0)`, a
  function of the CG shift and total thrust alone; the allocator does not appear.
  Measured, the two mixers agree to 2–8 %, never 5×. Reconciling the literal test
  would push `external_wrench` to use the mixer's belief instead of the nominal
  allocation — which makes every payload invisible, and is **the same
  substitution §9.4's check 3 warns about one level up.**
- **N-8** — the oracle observation channel handed ground truth to the controller
  even with an estimator attached. Invisible in any metric; the numbers simply
  come out better than they should.

Both were found by deriving the equation and measuring, not by reading.

[`AUDIT.md`](AUDIT.md) adds the deployment-side findings: PX4's `SIM_GZ_EC_MIN`
idle floor (`u_hover = 0.7287`, not 0.7694), the allocator's `CA_ROTORn_KM = 0.05`
against the rotors' 0.016 (a real 3.125× yaw-authority error, now modelled), and
the `actuator_motors` PWM normalisation that the RDP's input frame must match.
`config/acmpc.yaml` now carries `om_min`, `k_m_ctrl` and `u_hover`, all asserted
against the study by `check_glue`.
