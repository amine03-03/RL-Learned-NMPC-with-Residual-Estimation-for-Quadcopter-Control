# The 17-state control model

The control model — what the NMPC/AC-MPC predicts with — was

```
x = [p(3) | v(3) | q(4)]                                      NX = 10, NE = 9
```

and is now

```
x = [p(3) | v(3) | q(4) | om(3) | Omega(4)]                   NX = 17, NE = 16
```

The input is unchanged: CTBR `u = [collective, om_cmd/om_max]`. Nothing about
the plant, the PX4 interface, the mixer or the RDP estimator moves. What changes
is that the rate loop and the actuator are **predicted** rather than assumed
instantaneous.

## Why

Three measurements, all against the 23-state plant, on the **nominal** plant
with zero wind (so every number below is structural model error, not a
disturbance), driven closed-loop by the study's own LQR anchor.

### 1. The old model's attitude prediction was wrong by degrees

Attitude RMSE of an open-loop prediction, circle @ 1.0–1.5 m/s:

| horizon | 10-state | +om (13) | **17-state** | oracle om |
|---|---|---|---|---|
| 1 step (20 ms) | 0.812° | 0.032° | **0.0029°** | 0.005° |
| 5 steps (100 ms) | 3.677° | 0.492° | **0.110°** | 0.023° |
| 10 steps (200 ms) | 5.666° | 1.140° | **0.307°** | 0.035° |

fig8 @ 3.0–4.0 m/s, 10 steps: 30.04° → 11.55° → **1.412°**.

`om` alone buys 25–33× at one step. `Omega` buys another 4–9× on top, and beats
an *oracle* that is handed the plant's true `om` — because the oracle still
computes thrust from `thrust_of(u)` while the 17-state model computes it from
`sum K_T Omega^2`, so it captures collective lag too.

The gain from `Omega` is **not** the rotor transient (that decays in ~2 control
steps). It is that `Omega` reproduces the real torque-generation path — the
mixer, the `k_m` 3.125× allocator mismatch (A3), and the `F_MAX`/`Om` saturation.
Those are standing errors, not transients, so they do not decay out of the
horizon.

### 2. The residual channel was mostly the autopilot's own lag

On a nominal plant with **no external wrench at all**, the old (4.9) residual
`om - om_cmd` measured:

| | circle/LQR | fig8/LQR | circle/PID |
|---|---|---|---|
| `\|\|om - om_cmd\|\|` | 0.788 rad/s | 3.405 | 0.069 |
| as a fraction of `\|\|om\|\|` | **97 %** | 94 % | 18 % |
| (4.13)'s image of the wrench | 0.0115 | 0.0330 | 0.0144 |
| ratio (4.9) / (4.13) | **69×** | **103×** | 4.8× |

The RDP predicts the wrench (object (b) of §4.3); `wrench_to_dmod` converted it
and fed `fc`. But what `fc` was actually wrong by (object (a)) was two orders of
magnitude larger, and almost all of it was the rate loop's tracking transient —
which the 10-state model had no way to represent. The adaptive rate channel
could address about 1 % of the error present.

With `om` a state, `fc` predicts the rate loop, so what is left in (4.9) is what
the nominal model genuinely fails to explain. And with a torque input the
conversion is **exact**: `alpha_res = J_nom^-1 tau`. `moment_gain`,
`DMOD_SETTLE_S` and the `first_order` / `closed_loop` modes are deleted —
`DMOD_MODES` is now `("exact", "none")`.

### 3. The old model over-claimed rate authority by 24×

`d(attitude)/d(u_roll)`, model against plant:

| horizon | plant | 10-state | 17-state |
|---|---|---|---|
| 1 step | 0.480 | 11.37 (**23.7×**) | ~1.0× |
| 10 steps | 75.75 | 113.2 (1.5×) | ~1.0× |

The plant's true one-step rate authority is 158× smaller than at 10 steps. The
10-state controller partly worked *because* it over-claimed, which acted as an
implicit gain.

### Under domain randomisation

The ranking survives; the margin narrows (attitude RMSE at 10 steps, circle):

| | 10-state | 13-state | 17-state |
|---|---|---|---|
| S1 nominal | 6.72° | 1.35° | 0.358° |
| S2 moderated | 23.75° | 9.08° | 3.09° |
| S3 raw OOD | 36.98° | 20.02° | 12.33° |

## What was deliberately left out

* **The rate integrator `I_om`** (16 states). Measured: 2.6 % better on a
  circle and 0.4 % *worse* on a fig8. It does not earn its states.
* **`K_d`** is dropped along with `K_i`, because they are a pair. The plant's
  integrator drives `om_f -> om_cmd`, so the plant's steady-state rate gain is
  exactly `om_max`. Keeping `K_d` without `K_i` leaves a 2 % droop
  (`om_ss = K_rate/(K_rate + K_d) om_cmd` = 9.804 instead of 10) — a *standing*
  bias that integrates straight into attitude error. Dropping both restores the
  exact DC gain and is 15–17 % better at a 0.2 s horizon.
* **The 40 Hz gyro filter** (`om_f ~ om`): 4 ms against `dt_c` = 20 ms.

So the inner loop in `fc` is a pure proportional law, `omdot = K_rate (om_cmd - om)`.

## Invariants that must keep holding

Both are pinned by tests, and both are *exact*, not approximate:

* steady-state `d a_z / d c = DAZ_DC_HOVER = 21.6670` — the (2.11) / A2 identity
* steady-state `d om_x / d u_roll = OM_MAX[0] = 10.000`

## Structural consequences

* `NE = 16`: `e = [dp | dv | 2 sgn q_ev | dom | dOmega / OM_SCALE]`.
  `OM_SCALE = 1000` is a pure change of coordinates — `err` and `e_to_state`
  divide and multiply by it symmetrically. Without it a channel of size ~769
  sits next to channels of size ~0.1 in a learned cost map whose entries already
  span five decades.
* `NTAU = 20`, so `REP_DIM["diag"]` goes 26 → 40 and the actor head grows with
  it. **Every checkpoint trained against the 10-state model is invalid.**
* `Omega_ref` comes from the nominal allocator at the reference wrench
  `[T_ref, tau_ref]`, with `om_dot_ref` from a centred difference of `om_ref`
  (itself a centred difference of `q_ref`). At hover `tau_ref = 0` and
  `T_ref = m g`, so `Omega_ref = OM_HOVER` exactly and `err()` vanishes on the
  trimmed hover state.
* `B_c` of the hover linearisation is **zero on every row but the rotor block** —
  the model is a cascade, `u -> Omega -> (v, om) -> ...`. C-2 therefore changes
  shape: `delta_dot = om` is now a statement about `A_c`.
* `step_c` takes `N_SUB_C = 2` RK4 sub-steps. The rotor pole (`tau_up` = 12.5 ms)
  is faster than `dt_c`; one step leaves 5.2 rad/s of `Omega` error against a
  converged reference, two leaves 0.6.
* `OBS_DIM` 40 → 47. `NOISE_LEVELS['w']` was declared but unreachable under the
  10-state model and is now a real gyro noise channel. The rotor block takes no
  sensor noise — `Omega` is not measured.

## Deployment: `Omega` is not telemetered

PX4 publishes no rotor speed. `ACMPCController` carries its own estimate,
propagated open-loop from the commands it has sent through the same `step_c` the
prediction uses. The mode is stable and fast, so it pulls in from a cold hover
start in ~4 control steps: measured 0.65 % of `OM_HOVER` on a gentle circle,
2.5–7 % on an aggressive fig8 (where the plant's rate integrator, which `fc`
drops, moves `Omega_cmd`). Even at the cold-start error the 17-state prediction
beat the 13-state one, so a coarse `Omega` is worth more than none.

## What still has to be re-run

Prediction accuracy is not tracking accuracy — AC-MPC *learns* its cost map and
can absorb some model error, so none of the ratios above transfer to closed-loop
RMSE unchanged. What is known so far, on a nominal circle at n = 32, T = 400:

| controller | RMSE | sat | crash |
|---|---|---|---|
| LQR | 0.0469 m | 0.000 | 0.000 |
| NMPC N=1 | 0.0481 m | 0.000 | 0.000 |
| NMPC N=5 | 0.0472 m | 0.000 | 0.000 |
| PID | 0.3702 m | 0.000 | 0.000 |

against the 10-state hand-tuned NMPC's reported 0.068 m. Still outstanding:

1. **Retrain Notebooks 3 and 5.** All AC-MPC checkpoints are invalid.
2. **Re-run the (Q_pos, R) grid of §8.1 step 6.** `Q_HAND`'s position/velocity/
   attitude block is carried over from the 10-state optimum; the rate and rotor
   blocks are *set, not searched*. No number in `Q_HAND` may be quoted as an
   optimum until that grid is re-run.
3. **`artifacts/` is stale** — every CSV there was produced by the 10-state model.
   `nb5_dmod_validity.csv` in particular should now read `om_ratio ~ 1` instead
   of the 69–103× shortfall, which is the headline result of this change.
