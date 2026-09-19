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

## 6. Known limits

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
