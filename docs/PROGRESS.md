# Project advancement — status to date

*Companion to `report/report.tex`. This file says **where the work is**; the
report says what it means. Every number below is from the `medium` scale and is
a pipeline measurement, not a reported result — the report's result tables stay
on `\BL` placeholders until the `full` campaign runs.*

---

## 1. The argument the project is built around

The report and the code follow one chain. It is worth stating once, because
every notebook exists to test one link of it.

1. **The problem.** Track an aggressive reference on a Holybro X500 whose plant
   is not the model: unknown payload, offset CG, slung load, wind.
2. **The classical answer works — and charges three times.** NMPC with a long
   horizon, and the LQR that anchors it, track accurately. They pay in
   *computation* (the horizon that buys accuracy buys the latency), in *model*
   (a multi-step solve compounds a wrong model where a fixed gain merely reacts),
   and in *cost* (13 hand-chosen weights that no part of MPC theory determines).
3. **So: shorten the horizon to one step, and put what the long horizon was
   doing into a learned object.**
4. **Increment 1 — learned local terminal cost (LLTC).** Put it in the terminal
   cost. *Measured: not enough.*
5. **Increment 2 — learned cost map (AC-MPC).** Put it in the whole cost map,
   trained by RL against the task, emitting the linear term as well as the
   quadratic one.
6. **Increment 3 — domain randomisation.** Generality bought with specificity:
   the averaging that makes the policy indifferent to *which* plant it is on
   makes it indifferent to the plant it is *actually* on.
7. **Increment 4 — adaptive AC-MPC.** Don't randomise less, observe more: a
   residual disturbance predictor (RDP) reconstructs the external wrench from a
   causal window of states and actuator commands.
8. **The endpoint: an enhanced cost and an enhanced model.** One-step
   optimisation, cost replaced by a learned map `θ`, dynamics replaced by
   nominal-plus-estimate `ψ`. The optimiser between them is unchanged, which is
   the point — hard constraints, explicit preview, horizon of one.

This chain is now written into the report in three places: *The argument, in one page*
(in **Structure of this report**), the *"What Increment N left unsolved"* bridges in the methodology
chapter, and **The endpoint: an enhanced cost and an enhanced
model** in the conclusion.

---

## 2. What exists

| Component | File | State |
|---|---|---|
| Batched double-precision X500 plant (23 states, rate loop, mixer, rotor lag, idle floor) | `study/x500_core_jax.py` | done |
| Differentiable iLQR layer (box-constrained, preview along the horizon, last-`k` differentiated) | `study/x500_core_jax.py` | done |
| LQR / NMPC / PID baselines | `study/x500_core_jax.py` | done |
| PPO with a KL backtracking line search, and TRPO | `study/x500_core_jax.py` | done |
| Adaptive env, 26-D causal frame, 4 RDP encoders (GRU/LSTM/TCN/CNN), 3 use-variants (A/B/C) | `study/adaptive_core_jax.py` | done |
| Notebooks 1–8 | `study/notebooks/nb*.py` | done; 1–7 run at `medium`, 8 partially |
| Figure rebuild from committed CSVs (F1–F31) | `study/make_report_figures.py` | done |
| PX4 / Gazebo ROS 2 workspace and the sim-to-real protocol | `ros2_ws/`, `docs/ROS2_WORKSPACE.md` | built, flights not yet flown |
| Report | `report/report.tex` | structure + narrative complete, results on `\BL` |

### The eight notebooks

| NB | Question | Figures |
|---|---|---|
| 1 | Baselines: does the online solve earn its computation? Horizon, noise, cost weights | F1–F5 |
| 2 | Increment 1 — LLTC: fit quality, horizon equivalence, in/out of distribution | F6–F8 |
| 3 | Increment 2 — AC-MPC: cost parametrisation, exploration, MPVE, PPO vs TRPO | F9–F13 |
| 4 | Increment 3 — domain randomisation and the conservatism premium | F14, F15 |
| 5 | Increment 4 — RDP in isolation and in closed loop, with the training-health gate | F16–F20 |
| 6 | The comparison: the ledger, degradation S1→S3 | F21–F24 |
| 7 | Flight visualisation, contact sheet, ground tracks (the only place `no_respawn=True`) | F25, F26 |
| 8 | Sensitivity and stress: plant mismatch, cost-weight stress, RDP window sweep, fragility | F27–F31 |

---

## 3. The debug campaign (G-series)

The first `medium` run produced tables that were **insensitive to whether
training happened at all** — `full/circle` read 0.9251 m with 0 landed policy
updates and 0.9252 m with 1538, because the 3 m respawn bound, not the
controller, set the number. Fifteen defects were found and fixed; all are
written up with measurements in `docs/AUDIT.md`.

The four that actually moved the results:

- **G1** — the PPO trust region never let a single step land. Rewritten as a
  backtracking line search that restarts from the same base each trial and keeps
  the critic step when the actor step is rejected.
- **G2** — the reward paid the policy to fly out of the ball (`fail` now covers
  `bad | spin | far`, not `bad | spin`).
- **G11** — `loss_fn` and its gradient were defined *inside* the iteration loop
  and recompiled every iteration. This, not the physics, was the wall-clock
  wall; it is also what made three earlier runtime estimates wrong.
- **G15** — the cost-map head initialisation did not scale with fan-in
  (`0.1` → `0.1/√hid`).

Also corrected along the way: two proposed saturation mechanisms that a table
refuted, one wrong claim about the 5 % saturation gate (`7f43343`), the
withdrawn G14 mechanism, a self-inflicted LLTC regression bisected across three
changes (`G7b/c/d`), and a relative-only mismatch metric that flattered the
worse controller by ≈ 4× (fixed by reporting the absolute crossover beside it).

15 regression tests guard the G-series fixes: `study/tests/test_g_training_regressions.py`.

---

## 4. What the `medium` run currently says

**Read as diagnostics.** `medium` is roughly 1/8 of the `full` budget; the
report's tables are deliberately still empty.

**The study's central claim reproduces.** AC-MPC at `N=1` with **0** tuned
quantities reads 0.1940 / 0.3859 / 0.6280 m (S1/S2/S3) against NMPC `N=1` with
**13** tuned quantities at 0.1359 / 0.2968 / 0.5014 m, at the same ≈ 3 ms
latency. The gap is real and is the honest headline: *a trade, not a win* — and
NB1's `(Q, R)` sweep puts the value of the human's weight choice at **10.1×**,
which is the axis the trade is being made on.

**Supporting positives.** `diag` beats `chol` and `full` 3/3, reproducing the
published finding. A model-free MLP with the same reward fails at 1.34 m.
Without respawn (NB7) AC-MPC holds 0.0499 m over 6.30 of 6.28 rad.

**Honest negatives, all of which stay in the report.**

- **LLTC does not achieve horizon equivalence.** Ratio LLTC `N=1` / NMPC `N=1`
  is 7.99 / 8.20 / 2.49 on circle / fig8 / square; 1.0 would be the claim.
- **The adaptive arm does not beat AC-MPC.** Worse on S1 and S2; better on S3 by
  0.0088 m against a seed spread of 0.0216 m — i.e. not a result. 8 of NB5's 9
  checks are FALSE, and the training-health precondition still fails
  (0.0645 / 0.0884 against the 5 % gate).
- **MPVE and PPO-vs-TRPO** are both below the seed spread.
- **NMPC `N=1` is *not* uniformly more fragile than LQR.** On the absolute
  crossover it loses to LQR only **above 1.20× mass and 1.30× inertia**, never
  below. The relative break factor says otherwise and is the metric that
  flatters the worse controller.
- **A longer RDP window does not buy accuracy** at this budget: `R²` peaks near
  `H = 16–32` and degrades by `H = 128`, where the TCN fit diverges outright.
  Since `H` is also the post-respawn warm-up (320 ms → 2.56 s), the shortest
  adequate window is the one to deploy.

---

## 5. Open items

**Blocking the report's result tables**

1. Run all eight notebooks at `X500_SCALE=full`, then `make_report_figures.py`.
2. NB8 §3 (F29, learned arms against plant mismatch) needs the `acmpc` and
   `acmpc_adaptive` checkpoints present; it currently skips.

**Known-open technical questions**

3. The NB5 training-health gate still fails. The candidate lever is
   `WRENCH_DR_START` 0.10 → 0.05 — untested.
4. `chol` / `full` initialise to `AAᵀ ≈ 3.16·I` against `diag`'s ≈ 1.60 / 1.78;
   the initialisation argument of the report predicts they need longer rather
   than being incapable, and the present sweep budget cannot separate the two.
5. No real flights yet. The ROS 2 workspace and protocol exist
   (`docs/ROS2_WORKSPACE.md`); the thrust-curve calibration is the first step.

---

## 6. Reproducing

```bash
cd study

# tests (ROS 2's launch_testing plugin conflicts with pytest here)
python -m pytest tests -q -p no:launch_testing

# the campaign, in order; medium ~ hours, full ~ 8x that
export X500_SCALE=medium          # smoke | medium | full
for n in notebooks/nb?_*.py; do python "$n" || break; done

# every figure, from the committed CSVs, no training
python make_report_figures.py
python make_report_figures.py --list      # what is buildable right now
```

`X500_ARTIFACTS` relocates the artifact root. A figure whose companion CSV is
absent is **skipped and named** — `report.tex`'s `\realfig` then renders a
labelled placeholder frame, because a silently invented figure is worse than a
visible gap.
