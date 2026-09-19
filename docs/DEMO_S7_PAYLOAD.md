# S7 — hover stabilisation under an asymmetric payload

Three controllers hold the same hover setpoint with the same 300 g payload
bolted 12 cm forward and 6 cm left of the centre of gravity. The comparison is
plotted live in the **Z–X plane**, and for the adaptive controller the **mass**
and **moment** estimates are plotted beside it.

| controller | solver | cost | sees `d_hat`? |
|---|---|---|---|
| `nmpc1` | iLQR, `N_p = 1` | hand-tuned `Q_HAND`/`R_HAND` | no |
| `acmpc` | iLQR, `N_p = 10` | learned cost map | no |
| `acmpc_adaptive` | iLQR, `N_p = 10` | learned cost map | **yes** — RDP residual in the prediction model |

Exactly one controller is fed the estimate. That routing lives in one table,
`controller_node.CONTROLLERS`, and `check_glue` asserts that the list of
controllers with `use_d` is exactly `["acmpc_adaptive"]` — so the label on the
plot cannot disagree with the wiring behind it.

---

## 1. The payload

`tools/payload_sdf.py` attaches a rigid link to the Gazebo x500 at a body-frame
(FLU) offset. The physics engine then produces the composite mass, the shifted
CG and the parallel-axis inertia — all of it, with nothing to keep in step at
run time.

```
m_p     = 0.300 kg                 14.53 % of the nominal 2.0643 kg
r_p     = (+0.12, +0.06, −0.04) m  body FLU
F_z     = −2.9420 N                residual of the NOMINAL-mass model
tau     = (−0.1765, +0.3530, 0) N m
|tau_y| = 29.1 % of the hover-constrained pitch authority (1.2137 N m)
dJ      = diag(0.00156, 0.00480, 0.00540) kg m²
```

**Why 29 %.** At hover each rotor sits at 5.061 N with 3.488 N of headroom
before `F_max`, so the largest pitch moment available without dropping out of
hover is `2 × 0.174 × 3.488 = 1.2137 N m`. A payload drawing 29 % of that leaves
the three controllers room to differ. One drawing 90 % would saturate all three
and the experiment would measure the allocator.

**Why patch the model rather than inject a wrench.** A wrench applied to the
base link reproduces the payload's *weight* but not its *inertia*, and it needs
the `apply-link-wrench` system plugin, which the stock PX4 worlds do not carry.

The `disturbance_manager` does **not** apply the payload; it publishes the
analytic truth on `/disturbance/ground_truth` for the logger and the plots. For
S7 alone it subscribes to odometry, because the weight is fixed in the **world**
frame while the moment it makes is taken in the **body** frame — so the truth
depends on attitude and is not a constant.

---

## 2. Mass and moment estimation

The RDP predicts a 6-D residual wrench `[F_ext, tau_ext]`, **not** a mass. The
moment estimate *is* `tau_ext`; there is nothing to derive. The mass is derived,
and `visualization/estimates.py` keeps the derivation in one place with the
assumption attached:

```
(9.9)   F = −T z_b (m_p / m)                      exact
(9.10)  m_hat = m_nom − F_z / g                   hover form, the default
(9.11)  r_x = +tau_y / (m_p g),  r_y = −tau_x / (m_p g)
```

**Tilt does not bias (9.10).** In a steady tilted hold `T cos(tilt) = m g`
exactly, so `F_z = −m_p g` at any tilt. What biases it is **vertical
acceleration**: `T cos(tilt) = m (g + a_z)` makes (9.10) read `m_p (g + a_z)/g`,
+10.2 % per 1 m/s² of climb. A hover hold is the benign case. When
`/fmu/out/actuator_motors` is available the plotter computes thrust from it and
uses the exact inversion of (9.9) instead, which is unaffected; the panel says
which mode it is in.

Only `r_x` and `r_y` are identifiable — a payload directly below the CG makes no
moment at hover, so `r_z` does not appear in (9.11) and is not reported. Below
30 g of estimated payload the ratio is noise over noise and the offset is
returned as `NaN` rather than as a large number.

---

## 3. PX4 message versioning

PX4 1.16 versions its uORB messages and the bridge advertises the version on the
wire (`/fmu/out/vehicle_odometry_v1`). **Every failure mode of a mismatch is
silent:**

| mistake | symptom |
|---|---|
| subscribe the unversioned name on a versioned graph | callback never invoked; the node sits on `state is None` and logs nothing, because waiting looks like starting up |
| publish the unversioned name | PX4 never sees a setpoint; offboard is rejected after 0.5 s; the vehicle never arms, or drops out mid-flight |
| right name, wrong ABI (`px4_msgs` from another tag) | deserialises into plausible-looking garbage — the worst of the three, because the run *looks* alive |

`acmpc_controller/px4_topics.py` therefore hard-codes nothing. It reads
`get_topic_names_and_types()`, matches `^<base>(_v<digits>)?$`, imports the type
string **the graph reports**, and picks the highest version whose type can
actually be imported. A name that is present but whose type cannot be imported
raises `VersionMismatch` with the wire type, the installed classes and the fix.

Every node resolves this way — controller, estimator, disturbance manager and
plotter — and the controller keeps the resolved *classes*, so nothing later
re-imports an unversioned class behind the resolution's back.

```bash
ros2 run acmpc_controller check_px4                    # census + resolution
ros2 run acmpc_controller check_px4 --namespace /px4_1 # multi-vehicle
```

The driver script runs this before flying and refuses to fly if it fails.

---

## 4. Running it

```bash
# 0. build
cd ~/RL-Learned-NMPC-with-Residual-Estimation-for-Quadcopter-Control
git pull
cd rdp_acmpc_ws
source /opt/ros/humble/setup.bash && source ~/px4_ws/install/setup.bash
colcon build --symlink-install && source install/setup.bash

# 1. attach the payload, then RESTART SITL (Gazebo caches models)
python3 tools/payload_sdf.py --install
python3 tools/payload_sdf.py --status

# 2. three terminals
MicroXRCEAgent udp4 -p 8888
cd ~/PX4-Autopilot && make px4_sitl gz_x500
ros2 run acmpc_controller check_px4        # must PASS before flying

# 3. fly all three, 40 s each, one accumulating figure
./tools/run_hover_payload_demo.sh

# or one controller at a time -- naming one REUSES the newest session, so
# these three commands build the same figure as the line above
./tools/run_hover_payload_demo.sh nmpc1
./tools/run_hover_payload_demo.sh acmpc
./tools/run_hover_payload_demo.sh acmpc_adaptive

# two of them, in order; --new-session forces a fresh directory
./tools/run_hover_payload_demo.sh nmpc1 acmpc_adaptive --new-session

# the launch file directly, if you want to watch one leg by hand
ros2 launch acmpc_controller hover_payload_demo.launch.py controller:=acmpc_adaptive

# re-render a finished session without a graph
ros2 run visualization live_zx --replay runs/demo_s7_<stamp> \
    --payload 0.30 0.12 0.06 -0.04

# 4. take the payload back off
python3 tools/payload_sdf.py --revert
```

Each session directory holds `zx_traces.npz`, `zx_metrics.json`,
`R-F13_zx_demo.png` and one `.log` per leg.

**The legs are sequential, not parallel.** One controller can fly a given SITL
at a time, so the traces accumulate: each leg is appended to `zx_traces.npz` and
reloaded at start-up, and the figure shows the finished controllers as static
curves beside the one currently flying. Loading never overwrites the controller
in flight. (If you genuinely want them simultaneous, that needs three namespaced
SITL instances — every node takes `px4_namespace`, and `check_px4` takes
`--namespace`, so the plumbing is there; the driver script does not do it.)

**Between legs the vehicle lands.** `timeout --signal=INT` ends a leg, the
setpoint stream stops, and PX4's offboard failsafe lands and disarms. That is
deliberate: each leg then takes off from the same state, so the three traces
share a starting condition instead of each beginning wherever the previous one
left off. `SETTLE` (default 20 s, `--settle`) is what that landing needs — it is
not padding.

---

## 5. Reading the figure

```
+---------------------------+------------------+
|                           |  m_hat(t)        |  adaptive only
|   Z-X plane               +------------------+
|   (altitude vs x)         |  tau_hat(t)      |  adaptive only
+---------------------------+------------------+
|   z(t), all three, with the hold setpoint     |
+-----------------------------------------------+
```

The Z–X plane is the headline because the payload's signature is a **coupled**
error: the standing pitch moment pushes the vehicle along +x while the extra
14.5 % of weight pulls z down, so the deviation is a diagonal excursion from the
setpoint, not an altitude drop. Plotting `z(t)` alone would hide half of it.

Reported per controller as the suptitle and in `zx_metrics.json`: `z_bias`,
`x_bias`, `z_rms`, `p_rms` (means over the **last 5 s**, not RMS from `t = 0`)
and `p_max`. The payload is bolted on for the whole episode, so what separates
the controllers is the offset they settle to; an RMS from `t = 0` buries it
under the common take-off transient.

The dashed lines on the mass and moment panels are the scenario's analytic
truth. `check_glue` asserts that those lines and the plant's payload come from
the same two numbers, so a drift between the SDF patch and the scenario cannot
silently make every plotted truth wrong.

---

## 6. What the controller is actually handed

Two things about the learned controller are easy to get wrong and silent when
you do.

**The observation is not decoration.** The cost map is a *function* of the
47-channel observation of (5.3):

```
e (16) | 3 x [p_ref(t+h) - p, v_ref(t+h)] (18) | int_ep (3) | du_bar (4)
       | ev_bar (3) | om_bar (3)
```

Four of those blocks carry **memory** — an integral of the position error and
three exponential means — which the controller advances every step exactly as
the training environment does. A controller that passes zeros, or that never
advances the memory, evaluates the learned cost at a point that never occurs
in training. Nothing raises: the widths match.

**Checkpoints do not all want the same width.** Variant C was trained with the
residual routed into the observation as well as the model, so its actor expects
`OBS_DIM + 6 = 53`. Handing it 47 is a broadcasting error deep inside the cost
map. The controller reads the width off the checkpoint's own normaliser and
builds to match, appending the estimate as the raw wrench or as the converted
residual according to the training config's `oracle_target` — the two have the
same width, so choosing wrongly would be silent too.

**The horizon is not yours to choose either.** The cost map's head emits one
parameter block per stage from a single dense layer, so its output width is
`N_train x REP_DIM` and `N` is fixed at training time. The `horizon` launch
argument is therefore a *request*: `nmpc1` flies at `N = 1` because that is what
its name means, and the learned controllers fly at whatever their checkpoint was
trained at, logging the fact when the two differ. Asking an `N = 1` head to fill
ten stages fails as `cannot reshape (1, 40) into (1, 10, 40)`, a long way from
the parameter that caused it.

The checkpoint preference is **B, then C, then the plain Notebook-3 model**, and
the node logs which it loaded. B routes the residual into the model only and
wins the scenario sweep; C routes it into both; the plain model is not adaptive
at all and is the last resort.

## 7. The first solve is compiled before the loop starts

JAX traces and compiles on the first call. Measured on a CPU-only `jaxlib`
that is **13.5 s**, against a steady-state **7.4 ms**. Paying it inside the
first timer callback blocks the executor, so no `OffboardControlMode` is
published while it happens — and PX4 refuses to enter offboard, or drops
straight out of it, if that stream stops for more than 0.5 s. The vehicle would
never arm, and the log would show a controller that looked perfectly healthy.

So `ACMPCController.warmup()` compiles against a synthetic hover during node
construction, where 13 s costs nothing, and discards the state it produces —
warm-up must not seed the rotor observer or (5.3)'s memory with a fiction. The
node logs the compile time. After it, the first real tick is 8.5 ms.

If you have a CUDA `jaxlib`, this is much faster; the warm-up is cheap either
way and the ordering is what matters.

## 8. Arming is confirmed, not assumed

The node used to send the two offboard-handshake commands once and then set
`armed = True` on the strength of having sent them:

```
_cmd(VEHICLE_CMD_DO_SET_MODE, 1.0, PX4_CUSTOM_MAIN_MODE_OFFBOARD)
_cmd(VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)
self.armed = True                       # <- a hope, not a fact
```

PX4 can refuse either one, and routinely does: the EKF has not converged, a
preflight check is failing, or it has not yet seen enough of the
`OffboardControlMode` stream to accept mode 6. Nothing came back, so the node
carried on publishing rate setpoints at a vehicle that was sitting disarmed on
the ground for the whole 40 s. Every node reported healthy, the solver hit its
timing budget, and the Z-X plot was a flat line at z = 0 — which looks exactly
like a controller that computes nothing.

The node now resolves `/fmu/out/vehicle_status` through `px4_topics` (the
user's graph carries it as `_v4`, so a hard-coded name would have produced a
subscription that is silently never called — the failure this module exists to
prevent) and reads the outcome back. Until PX4 reports **both**
`arming_state == ARMING_STATE_ARMED` **and** `nav_state ==
NAVIGATION_STATE_OFFBOARD`, the request is re-sent every `arm_retry_s` (default
0.5 s), asking only for the half that is still missing, and the console says
what is being withheld:

```
PX4 has not accepted the handshake after 4 request(s): arming_state=1
(want 2), nav_state=4 (want 14).  Nothing will move until both match --
the PX4 console carries the rejection reason.
```

Both halves are checked because armed-but-not-offboard is the trap. PX4 is then
flying under its own controller and discarding every setpoint published here,
so the trace is flat for a reason that has nothing to do with the controller
under test.

The two codes are taken off the resolved message class when it carries them as
constants, falling back to the literals 2 and 14 only when it does not — the
same reason the topic name is read off the graph rather than written down.

The watch does not stop at takeoff. If PX4 leaves armed/offboard mid-run — a
failsafe, or the land detector deciding an asymmetrically loaded vehicle has
touched down — the node logs an error naming the instant, because everything
in the trace after that point is PX4 flying, not the controller being measured.

When `vehicle_status` never appears on the graph at all there is nothing to
read back, and the node falls back to the old open-loop behaviour with a
warning rather than refusing to fly.

## 9. Known limits

* **`actuator_motors` is not optional for the moment channels.** Per §6.2 the
  PWM block is the only observable channel for a *standing* moment: an
  integrating rate loop drives a constant external moment out of the rate error
  while the mixer output holds its spread indefinitely. The estimator refuses to
  start without it by default (`require_actuator_motors`).
* The RDP was trained on position hold (E-0), which is what this demo flies — so
  this is in-distribution for the estimator and not a generalisation claim.
* `m_hat` is a derived quantity under (9.10) or (9.9), not a network output. The
  panel labels which.
* The three legs are flown back to back in one SITL session; they are not
  simultaneous and are not seed-matched to each other beyond flying the same
  scenario and setpoint.
