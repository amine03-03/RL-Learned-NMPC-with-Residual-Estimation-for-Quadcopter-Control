# Adaptive AC-MPC for a quadrotor, with a residual dynamics predictor

An implementation of `AGENTS_SPEC_Adaptive_ACMPC.md`: a model predictive
controller whose **stage cost is learned** rather than tuned, extended with a
causal **residual dynamics predictor** (RDP) that estimates the disturbance
online, plus the PX4/Gazebo workspace that flies it.

The licence for the idea is Gros & Zanon (2020): for a *wrong* model, the
finite-horizon problem solved under that model returns the optimal policy of the
true system once its stage and terminal costs are adapted. Equation (1.1) is not
computable, so it is learned. That is the whole project.

---

## Read this first: eight specification errors

The specification says *"never let a claim outrun a measurement"*. Applying that
to the specification itself, **eight of its statements do not survive checking**.
Each is recorded in [`docs/CORRECTIONS.md`](docs/CORRECTIONS.md) with the
derivation and the measurement that refutes it, and each is asserted by a test in
`study/tests/test_spec_corrections.py`, so a correction cannot quietly rot.

| id | what the spec says | what is true | how it was caught |
|---|---|---|---|
| **C-1** | (5.6) `A_c` attitude block is `−g·Ξ` | `+g·Ξ` | the spec's sign gives an LQR gain with closed-loop spectral radius **1.1807 — unstable** |
| **C-2** | (5.6) `B_c` attitude block is `½·diag(ω_max)` | `diag(ω_max)` | autodiff gives (10, 10, 4), not (5, 5, 2); the ½ is already consumed by `e₇..₉ = 2q_v` |
| **C-3** | (6.1) `F_ten = m_ℓ(g·n̂·e₃ + L‖ṅ̂‖²)` | the sign of the gravity term is `−` | as written, a hanging load pushes the airframe **up** (−3.0163 N against +3.0163 N) |
| **C-4** | §7.2 suites are "disjoint on drag, lag, thrust and wind" | disjoint on drag, lag and **wind only** | thrust, mass, `K_w` and inertia all overlap: the raw S3 band is a superset, and contraction preserves inclusion |
| **C-5** | §4.2 the (4.5) sign slip gives `κ_a = 4.6464` | it gives **6.8602** | dense sampling; the `+` sign and `κ_a = 9.077877` are confirmed |
| **C-6** | T-12 accepts on `‖τ_ext‖` being 5× larger under the nominal mixer | `τ_ext` is the **one quantity the mixer does not change** | at equilibrium it reduces to `−m_p g(0.174,−0.174,0)`; measured, the mixers agree to 2–8 %. What differs without bound is the rate-loop effort |
| **C-7** | §5.9 maps the cost range linearly, starting at ≈5×10⁴ | that is **660× the terminal matrix** | at `N=1` the solver returns `\|δu\| = 1.9e-5`: AC-MPC degenerates to pure feed-forward and **every row of every sweep in Notebook 3 came out identical** |
| **N-7** | §6.1's `asym` levels go to `f = 0.11` | the trim limit is **`f = 0.0812`** | holding the standing moment needs `J·K_i·I_lim = 0.286 N·m`; at 0.11 the integrator pins at 3.00, tilt reaches 48° and the vehicle cannot hover |

C-1 and C-7 are the two that matter most: the first makes the classical baseline
unstable, the second makes the learned controller inert while raising no error
anywhere. Both were found by measurement, not by reading.

Four implementation bugs of the same character were found and fixed while
testing; they are listed at the end of this file.

---

## Layout

```
study/                     pure JAX, no ROS, no PyTorch, float64 throughout
  x500_core_jax.py         §2-§5  model, error coordinates, iLQR, Env, PPO/TRPO
  adaptive_core_jax.py     §6     scenarios, the four RDP encoders, AdaptEnv
  study_prelude.py         §7.1   scale table and notebook furniture
  study_moderate.py        §7.2   disturbance moderation, the fixed-pilot start
  viz.py                   §7.3   flight recording, clips, the manifest
  export_estimator.py      §9.5   the export bridge (fails on parity failure)
  rdp_infer.py             §9.5   pure-NumPy forward pass, shipped to the node
  check_consistency.py     §11    study <-> workspace constants
  tests/                   §11    T-1..T-20 and every correction above
  notebooks/nb1..nb7       §8     percent-format sources; build_notebooks.py -> .ipynb
rdp_acmpc_ws/              §9     ROS 2 / PX4 / Gazebo
  src/acmpc_controller/      control node, PX4 frames, B_d bridge, preflight
  src/rdp_estimator/         ring buffer, NumPy inference, watchdog, timing
  src/disturbance_manager/   S0..S6, publishes ground truth (evaluator only)
  src/reference_generator/   Lissajous (9.5) with its feasibility gate
  src/experiment_manager/    E-A..E-D sweeps, run dirs, config capture
  src/state_logger/          50 Hz synchronised CSV
  src/visualization/         live panel R-F1 and offline R-F2..R-F12
  config/                    acmpc, rdp, disturbances, experiments
artifacts/                 every CSV, figure and checkpoint the notebooks write
docs/CORRECTIONS.md        the eight corrections, with their measurements
docs/ROS2_WORKSPACE.md     what is in the workspace, why, and what is not flown
```

[`docs/ROS2_WORKSPACE.md`](docs/ROS2_WORKSPACE.md) covers the deployment half in
detail: the pure-core / thin-wrapper split and the reason for it, how causality
is enforced structurally rather than by convention, each package and the trap it
guards, the two check binaries with their measured results, and an explicit list
of what is verified here against what needs a live PX4/Gazebo graph.

## Running it

```bash
pip install numpy scipy jax jaxlib optax pandas matplotlib pytest pyyaml nbformat

# the test suite -- every one of these exists because the error it catches is silent
cd study && python -m pytest tests/ -x -q && python check_consistency.py

# the notebooks, in order (each consumes the ones before it)
export X500_SCALE=smoke          # smoke | medium | full
cd notebooks && for n in nb?_*.py; do python "$n" || break; done
python build_notebooks.py        # -> .ipynb

# the deployment bridge, then the workspace checks
cd .. && python export_estimator.py --arch GRU --out ../rdp_acmpc_ws/models/rdp_gru.npz
cd ../rdp_acmpc_ws && colcon build && source install/setup.bash
ros2 run acmpc_controller check_ctbr && ros2 run acmpc_controller check_glue
```

### Scale discipline — read before quoting any number

`X500_SCALE` selects §7.1's table. **`smoke` produces undertrained policies by
design**; its learned-controller numbers are not results, and the reporting layer
detects the signature (hand-built controllers unaffected, every learned
controller diverging) and prints a warning naming the table. `medium` is an
addition of this repository, for verifying a build without paying for `full`.

**The artefacts committed here are from a `smoke` run.** They demonstrate that
the pipeline executes end to end and that every gate fires when it should; they
are not the study's results. §13 requires that every learned-controller number in
a write-up come from a `full` run.

Three gates fire at `smoke`, correctly, and each reports rather than hides:

- Notebook 2's fit gate: `R² = −0.80 ≤ 0`, so the terminal-cost fit is worse than
  predicting a constant and the horizon-equivalence claim is reported as
  **unsupported** — not tuned until it passes.
- Notebook 4 reports the conservatism premium **sign first** (−0.7 %, negative:
  no premium in this run) and gives both explanations two seeds cannot separate.
- Notebook 5's precondition gate: the Oracle arm saturates at 14.5 % against the
  5 % limit, so the closed-loop numbers are declared **not readable** as a
  decomposition into value-of-information and cost-of-estimation.

## What the pipeline produces at `smoke`

Hand-built controllers are scale-independent, so these are meaningful:

- **§2 constants** reproduce to 1e-6, all derived from the SDF values, none pasted.
- **T-8**, the most valuable test: residual and wrench are **identically zero**
  (< 1.2e-16) on the undisturbed plant.
- **Reference feasibility**: the superellipse at R = 0.5 demands 40.85 m/s²
  uncapped against a 13.35 m/s² envelope — 3.1×. With (4.6) no path exceeds
  8.0098 m/s².
- **The preview is not optional**: with the reference frozen, error grows with the
  horizon on every smooth path (circle 0.078 → 0.411 m over N = 1 → 20) while the
  preview column falls and saturates.
- **Weight tuning moves the answer** by 9× over the admissible (Q, R) grid — the
  measurement that motivates the whole study.
- **Does the online solve earn its compute?** On smooth paths, barely: LQR is
  6–8 % *better* than NMPC N=1. On the superellipse it is 78 % worse. That split
  is the honest answer to §8.6's first question.
- **The learned cost removes all 13 hand weights.** `diag` wins 3/3 tasks (the
  published finding reproduces) and AC-MPC reaches 0.137 m against tuned NMPC's
  0.111 m with `tuned = 0`.
- **The RDP is admissible.** All four encoders run under 2 ms single-window in
  NumPy against the 20 ms period, with JAX↔NumPy parity ≤ 1.1e-14.

## Things this implementation is careful about

- **Latency has one definition.** `ms_per_step / batch` is throughput; only
  `solve_latency_ms`, on a batch of one, may be compared to 20 ms.
- **Ground truth never reaches the controller.** The RDP's window comes from a
  ring buffer the disturbance manager cannot write to. The oracle observation
  channel is routed through `d_channel()`, so an attached RDP makes it carry the
  *prediction* — the earlier version handed truth to the controller, and a
  regression test now pins the behaviour.
- **Saturation is reported beside every RMSE.** When a controller is pinned
  against its input box the path has stopped mattering and the number describes
  the disturbance.
- **`T_max` is derived in one module** and every consumer is asserted against it;
  `check_glue` and `check_consistency.py` are deliberately independent so one
  edit cannot satisfy both.
- **Windows never straddle an episode boundary**, and splits are by complete
  episode.
- **Adaptation time is bounded by the disturbance-off instant.** Without that, a
  predictor stuck at zero "converges" the moment the disturbance stops and scores
  7.0 s instead of the NaN it deserves.

## Bugs found while testing

| where | bug | consequence |
|---|---|---|
| `lin_traj` | differentiated the batched map | built a (B,9,B,4) cross-Jacobian, B× too large and zero off the diagonal |
| CNN encoder | decimated with `x[:, ::2]` | dropped the **most recent** frame at even lengths — the predictor was blind to the sample it must react to |
| `Env` | applied `fixed=` overrides *after* ω was derived from R | a controlled experiment silently varied the quantity it pinned |
| `Env._oracle` | returned `d_truth()` with an RDP attached | every "RDP" row was secretly an oracle row |

## References

The full bibliography is §12 of the specification. The load-bearing ones are
Gros & Zanon (IEEE TAC 2020, arXiv:1904.04152) for the licence to learn the
cost; Romero, Song & Scaramuzza (ICRA 2024, arXiv:2306.09852) for the
cost-map-over-short-MPC architecture; Büskens & Maurer (2001) for why the
parameter gradient is a by-product of the solve; and Li & Todorov (2004) with
Tassa et al. (2012) for the iLQR and its regularisation. The identifier
`arXiv:2605.16015` that the deployment specification cites has **not** been
verified here and is not relied on anywhere in this code.
