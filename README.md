# RL-Learned NMPC with Residual Estimation for Quadcopter Control

A model predictive controller for a quadrotor (PX4 **Holybro X500**) whose cost is
**learned** instead of hand-tuned, extended with a causal **residual dynamics
predictor (RDP)** that estimates the disturbance online. It also contains a
standalone study of the **learned Lyapunov terminal cost (LLTC)** that lets a
one-step NMPC replace a long-horizon one.

The project has three parts:

| part | folder | what it does | needs |
|---|---|---|---|
| **1. Study** | `study/` | model, iLQR-MPC, LLTC, AC-MPC (RL-learned cost map), domain randomisation, RDP; tests and 8 notebooks | Python + JAX |
| **2. LLTC hover study** | `lltc/` | one script: LLTC-NMPC (N = 1) vs NMPC (N set by you) on hover, six figures | Python + CasADi + PyTorch |
| **3. Deployment** | `rdp_acmpc_ws/` | ROS 2 workspace that flies the controller on PX4 SITL + Gazebo | Ubuntu 24.04, ROS 2 Jazzy, PX4, Gazebo Harmonic |

Parts 1 and 2 run on any Linux/macOS machine with Python ≥ 3.11. Part 3 needs
the ROS 2 / PX4 / Gazebo stack described in [§5](#5-part-3-ros-2--px4--gazebo-simulation).

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Installation (Python)](#2-installation-python)
3. [Part 1: the study (`study/`)](#3-part-1-the-study-study)
4. [Part 2: LLTC hover study (`lltc/`)](#4-part-2-lltc-hover-study-lltc)
5. [Part 3: ROS 2 / PX4 / Gazebo simulation](#5-part-3-ros-2--px4--gazebo-simulation)
6. [Configuration files](#6-configuration-files)
7. [Documentation index](#7-documentation-index)
8. [References](#8-references)

---

## 1. Repository layout

```
.
├── README.md                    this file
├── requirements.txt             Python deps for study/ and the ROS 2 nodes
├── study/                       Part 1, pure JAX, float64, no ROS
│   ├── x500_core_jax.py           model, error coordinates, iLQR, Env, PPO/TRPO
│   ├── adaptive_core_jax.py       disturbance scenarios, the four RDP encoders, AdaptEnv
│   ├── study_prelude.py           scale table (smoke | medium | full) and helpers
│   ├── study_moderate.py          disturbance moderation, LLTC controller factory
│   ├── viz.py                     flight recording, clips, manifest
│   ├── export_estimator.py        trained RDP -> framework-free .npz (parity-checked)
│   ├── rdp_infer.py               pure-NumPy RDP forward pass used by the ROS node
│   ├── check_consistency.py       study <-> workspace constants check
│   ├── tests/                     pytest suite (120 tests, incl. the ROS-node wiring)
│   └── notebooks/                 nb1..nb8 as runnable .py (+ generated .ipynb)
├── lltc/                        Part 2
│   ├── lltc_hover.py              LLTC-NMPC vs NMPC on hover, CTBR, CasADi/IPOPT
│   ├── requirements.txt
│   └── README.md                  method, settings and results of the script
├── rdp_acmpc_ws/                Part 3, ROS 2 workspace (ament_python)
│   ├── src/acmpc_controller/      control node, study-exact reference/observation,
│   │                              PX4 frames, B_d bridge, /acmpc/status, preflight checks
│   ├── src/rdp_estimator/         ring buffer fed from /acmpc/status + actuator_motors,
│   │                              NumPy inference, watchdog, smoothing
│   ├── src/disturbance_manager/   scenarios S0..S6 -> ground truth + applied in Gazebo
│   ├── src/reference_generator/   Lissajous / hold reference with feasibility gate
│   ├── src/experiment_manager/    experiment plan and config capture
│   ├── src/state_logger/          50 Hz CSV logger (one row per controller tick)
│   ├── src/visualization/         offline figures from a run directory
│   ├── config/                    acmpc.yaml, rdp.yaml, disturbances.yaml, experiments.yaml
│   └── models/                    exported RDP weights (.npz), ready to use
├── artifacts/                   every CSV / figure the notebooks and lltc/ write
└── docs/
    ├── ROS2_WORKSPACE.md          design of the deployment workspace
    ├── CORRECTIONS.md             8 corrections to the build specification
    ├── AUDIT.md                   audit against the real PX4 x500 and source papers
    └── TROUBLESHOOTING.md         known JAX/XLA issue and its fix
```

---

## 2. Installation (Python)

### 2.1 System packages

Ubuntu 22.04 / 24.04 (Debian-like):

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip ffmpeg
```

`ffmpeg` is only used by Notebook 7 to write `.mp4` flight clips. Without it,
the notebook writes PNG frames instead.

JAX 0.10 needs **Python ≥ 3.11**. Ubuntu 24.04 ships 3.12. On 22.04, install
3.11 first:
`sudo apt install -y software-properties-common && sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt install -y python3.11 python3.11-venv`,
then use `python3.11` instead of `python3` below.

### 2.2 Clone

```bash
git clone https://github.com/amine03-03/RL-Learned-NMPC-with-Residual-Estimation-for-Quadcopter-Control.git
cd RL-Learned-NMPC-with-Residual-Estimation-for-Quadcopter-Control
export REPO=$PWD                      # used by every command below
```

### 2.3 Virtual environment for Part 1 (study)

```bash
cd $REPO
python3 -m venv .venv-study
source .venv-study/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import jax; jax.config.update('jax_enable_x64', True); print(jax.__version__, jax.devices())"
```

For an NVIDIA GPU, replace the `jax`/`jaxlib` lines with
`pip install "jax[cuda13]==0.10.2"`. Everything also runs on CPU.

### 2.4 Virtual environment for Part 2 (LLTC)

This is kept separate because PyTorch is large and the study does not need it.

```bash
cd $REPO
python3 -m venv .venv-lltc
source .venv-lltc/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch (~200 MB)
pip install -r lltc/requirements.txt
```

Leave a venv with `deactivate`.

---

## 3. Part 1: the study (`study/`)

```bash
cd $REPO && source .venv-study/bin/activate
```

### 3.1 Tests and consistency check (run these first)

```bash
cd $REPO/study
python -m pytest tests/ -q            # 120 passed, about 10-15 min on CPU
python check_consistency.py           # -> "all consistency checks passed"
```

### 3.2 Notebooks

The notebooks are stored as runnable percent-format `.py` files. They must run
**in order**, because each one consumes the artifacts of the ones before it:

| # | file | establishes | writes to `artifacts/` |
|---|---|---|---|
| 1 | `nb1_control_problem.py` | model self-test, LQR / NMPC baselines, horizon and weight sweeps | `common/`, `lltc/nb1_classical.csv` |
| 2 | `nb2_lltc.py` | learned terminal cost inside the iLQR-MPC | `lltc/model.pkl` |
| 3 | `nb3_acmpc.py` | AC-MPC: RL-learned cost map | `acmpc/model.pkl` |
| 4 | `nb4_domain_randomisation.py` | domain randomisation | `domrand/*.pkl` |
| 5 | `nb5_adaptive.py` | adaptive AC-MPC + RDP encoders | `acmpc_adaptive/` |
| 6 | `nb6_comparison.py` | all controllers on all scenarios | `common/ledger.csv` |
| 7 | `nb7_flight_visualisation.py` | flight clips | `videos/*.mp4` |
| 8 | `nb8_sensitivity.py` | sensitivity and stress | `sensitivity/` |

The scale is chosen with `X500_SCALE`:

| scale | purpose | time |
|---|---|---|
| `smoke` (default) | checks that the pipeline runs end to end. **Learned-controller numbers are not results** | minutes |
| `medium` | verifies a build | ~1–2 h |
| `full` | the only scale whose numbers may be quoted | ~10 h (GPU recommended) |

```bash
cd $REPO/study/notebooks
export X500_SCALE=smoke                     # smoke | medium | full
for n in nb?_*.py; do echo "== $n"; python "$n" || break; done
python build_notebooks.py                   # regenerate the .ipynb files from the .py sources
```

To run one notebook interactively instead: `pip install jupyter`, then
`jupyter notebook nb1_control_problem.ipynb`.

If JAX aborts with `tile size ... 3 % 4 != 0`, see
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). The quick workaround is
`JAX_PLATFORMS=cpu python <notebook>.py`.

### 3.3 Export the residual predictor for the ROS node (optional)

The RDP weights already exported are committed in `rdp_acmpc_ws/models/`. To
re-export from your own Notebook 5 run (this needs
`artifacts/acmpc_adaptive/estimators/all.pkl`):

```bash
cd $REPO/study
python export_estimator.py --arch GRU --out ../rdp_acmpc_ws/models/rdp_gru.npz
```

The export **fails** if the NumPy forward pass differs from the JAX one.

---

## 4. Part 2: LLTC hover study (`lltc/`)

This is a standalone re-implementation of *Abdufattokhov, Zanon & Bemporad,
"Learning Lyapunov terminal costs from data for complexity reduction in NMPC",
IJRNC 2024*. It covers hover stabilisation with collective-thrust/body-rate
(CTBR) inputs and a simple disturbance-free model.

```bash
cd $REPO && source .venv-lltc/bin/activate
python lltc/lltc_hover.py              # full run, ~15 min on a laptop CPU
python lltc/lltc_hover.py --quick      # smoke run, ~2 min (numbers are NOT results)
python lltc/lltc_hover.py --M 3000 --epochs 6000 --n-ic 6 --seed 1   # other settings
```

**Set the NMPC horizon** at the top of `lltc/lltc_hover.py`:

```python
N_MPC = 25            # <<< prediction horizon N of the baseline NMPC
```

Outputs go to `artifacts/lltc_hover/`:

| figure | content |
|---|---|
| `fig1_fit_quality.png` | R² of the learned cost-to-go, train / test |
| `fig2_altitude.png` | altitude tracking, LLTC vs NMPC |
| `fig3_control_effort.png` | thrust, body rates and total control effort, LLTC vs NMPC |
| `fig4_ood.png` | out-of-distribution initial states, LLTC vs NMPC |
| `fig5_weight_sensitivity.png` | sensitivity to the Lyapunov penalty λ and to the stage weights, LLTC only |
| `fig6_computation.png` | solve time (average, worst case, histogram, vs horizon), LLTC vs NMPC |

Each figure has a companion `.csv`, and `summary.csv` holds the headline
numbers. The method, every setting and the measured results are described in
[lltc/README.md](lltc/README.md).

---

## 5. Part 3: ROS 2 / PX4 / Gazebo simulation

**Tested stack:** Ubuntu **24.04**, ROS 2 **Jazzy**, Gazebo **Harmonic**, PX4
**v1.16.2**, `px4_msgs` **release/1.16**, Micro XRCE-DDS Agent **v2.4.3**. This
matches the PX4 recommendation for ROS 2. JAX 0.10 does not install on 22.04's
Python 3.10, so Humble on 22.04 is not supported here.

What follows uses three directories. Adapt the paths if you like.

| path | content |
|---|---|
| `~/PX4-Autopilot` | PX4 firmware + Gazebo SITL |
| `~/ros2_px4_ws` | `px4_msgs` + Micro XRCE-DDS Agent |
| `$REPO/rdp_acmpc_ws` | this project's ROS 2 packages |

### 5.1 Install PX4 and Gazebo Harmonic

```bash
cd ~
git clone -b v1.16.2 --recursive https://github.com/PX4/PX4-Autopilot.git
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh          # installs toolchain + Gazebo Harmonic
# log out and back in (or reboot) once, as ubuntu.sh asks
```

Stock PX4 v1.16.2 does **not** publish `/fmu/out/actuator_motors`, but the RDP
needs it: it is the only observable channel for a standing moment, and preflight
check 4 refuses moment scenarios without it. Add it to the uXRCE-DDS topic list,
then build:

```bash
cd ~/PX4-Autopilot
python3 - <<'EOF'
p = "src/modules/uxrce_dds_client/dds_topics.yaml"
s = open(p).read()
entry = "  - topic: /fmu/out/actuator_motors\n    type: px4_msgs::msg::ActuatorMotors\n\n"
if "/fmu/out/actuator_motors" not in s:
    s = s.replace("subscriptions:\n", entry + "subscriptions:\n", 1)   # append to publications
open(p, "w").write(s)
EOF
grep -n "fmu/out/actuator_motors" src/modules/uxrce_dds_client/dds_topics.yaml   # must print one line
make px4_sitl                                       # first build, ~5-10 min
```

### 5.2 Install ROS 2 Jazzy

```bash
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update && sudo apt install -y curl
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt update && sudo apt upgrade -y
sudo apt install -y ros-jazzy-desktop ros-dev-tools
sudo apt install -y ros-jazzy-ros-gzharmonic     # ros_gz_bridge + ros_gz_interfaces (disturbances)
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source /opt/ros/jazzy/setup.bash
```

### 5.3 Build `px4_msgs` and the Micro XRCE-DDS Agent

`px4_msgs` **must match the PX4 version** (release/1.16 for PX4 v1.16.x).

```bash
mkdir -p ~/ros2_px4_ws/src && cd ~/ros2_px4_ws/src
git clone -b release/1.16 https://github.com/PX4/px4_msgs.git
git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd ~/ros2_px4_ws
source /opt/ros/jazzy/setup.bash
colcon build                                        # ~5 min
source ~/ros2_px4_ws/install/setup.bash
ros2 interface show px4_msgs/msg/VehicleOdometry | head -3   # sanity check
```

### 5.4 Python environment for the nodes

The nodes run on ROS's Python (3.12) **and** import JAX (the controller uses
`study/x500_core_jax.py`). Create a venv that can also see the ROS packages.
Do **not** activate `.venv-study` here.

```bash
source /opt/ros/jazzy/setup.bash
cd $REPO
python3 -m venv --system-site-packages .venv-ros
touch .venv-ros/COLCON_IGNORE
source .venv-ros/bin/activate
pip install -r requirements.txt
pip install "setuptools<80"     # colcon --symlink-install fails with setuptools >= 80
python -c "import rclpy, jax, ros_gz_interfaces; print('rclpy + jax + ros_gz OK')"
```

### 5.5 Build this workspace

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_px4_ws/install/setup.bash
source $REPO/.venv-ros/bin/activate
cd $REPO/rdp_acmpc_ws
python -m colcon build --symlink-install
```

`python -m colcon` (instead of plain `colcon`) makes the node entry points use
the venv's Python, which has JAX. `--symlink-install` lets the controller find
`study/` relative to its source file.

### 5.6 Environment for every new terminal

Every terminal that runs a node needs the same environment. Create it once
(with `$REPO` set as in §2.2):

```bash
cat > ~/acmpc_env.sh <<EOF
export REPO=$REPO
source /opt/ros/jazzy/setup.bash
source ~/ros2_px4_ws/install/setup.bash
source \$REPO/.venv-ros/bin/activate
source \$REPO/rdp_acmpc_ws/install/setup.bash
export PYTHONPATH=\$REPO/study:\$PYTHONPATH
cd \$REPO/rdp_acmpc_ws        # nodes resolve config/ and models/ relative to here
EOF
cat ~/acmpc_env.sh             # the first line must show your absolute repo path
```

Then start every node terminal with `source ~/acmpc_env.sh`.

### 5.7 Offline preflight (no simulator needed)

```bash
source ~/acmpc_env.sh
ros2 run acmpc_controller check_glue        # constants + B_d bridge + .npz models -> "check_glue PASSED"
ros2 run acmpc_controller check_ctbr        # frames + CTBR mapping; checks 3-4 SKIP until a sim is up
ros2 run experiment_manager run_experiments # prints the 134-run experiment plan (dry run)
```

### 5.8 Run the simulation

Open **one terminal per step**.

**Terminal 1: DDS agent** (the bridge between PX4 and ROS 2)

```bash
source ~/ros2_px4_ws/install/setup.bash
MicroXRCEAgent udp4 -p 8888
```

**Terminal 2: PX4 SITL + Gazebo with the x500**

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500
```

Gazebo opens with the x500 (model `x500_0`, world `default`) on the ground, and
the `pxh>` shell appears in this terminal. SITL has no RC and no ground station,
so allow offboard flight without them (once; the values persist in the SITL
parameter file):

```
pxh> param set NAV_DLL_ACT 0
pxh> param set NAV_RCL_ACT 0
pxh> param set COM_RCL_EXCEPT 4
```

Check that PX4 reaches ROS. In another terminal,
`source ~/acmpc_env.sh && ros2 topic list | grep fmu` must list
`/fmu/out/vehicle_odometry`, `/fmu/out/actuator_motors` (only after the §5.1
patch), `/fmu/out/vehicle_status_v1` (PX4 v1.16 suffixes versioned messages),
`/fmu/in/vehicle_rates_setpoint` and `/fmu/in/offboard_control_mode`.

**Terminal 3: Gazebo wrench bridge** (lets `disturbance_manager` push forces
into Gazebo's `ApplyLinkWrench` system, which PX4's server.config loads)

```bash
source ~/acmpc_env.sh
ros2 run ros_gz_bridge parameter_bridge \
  "/world/default/wrench/persistent@ros_gz_interfaces/msg/EntityWrench]gz.msgs.EntityWrench" \
  "/world/default/wrench/clear@ros_gz_interfaces/msg/Entity]gz.msgs.Entity"
```

**Take off** (in the `pxh>` shell of terminal 2), and wait until it hovers (≈ 2.5 m):

```
pxh> commander takeoff
```

**Terminal 4: live preflight, including calibration**

```bash
source ~/acmpc_env.sh
ros2 run acmpc_controller check_ctbr --moment-scenario
```

Check 4 (`actuator_motors` publishing) must now **PASS**. For check 3, read the
hover motor command in `pxh>` with `listener actuator_motors` (the steady
`control[0..3]`, about 0.73) and pass it: `ros2 run acmpc_controller check_ctbr --y-measured 0.73`.

**Terminals 5–7: controller, estimator, logger**

```bash
# T5  controller.  Start it first: it JIT-compiles the solve (~10-30 s) and is
#     ready when it logs "mode=... N=...".  Until OFFBOARD it streams setpoints
#     that hold the current position; at OFFBOARD it flies a 4 s minimum-jerk
#     entry from where the vehicle is to the start of the path, then the path.
source ~/acmpc_env.sh && ros2 run acmpc_controller controller_node --ros-args --params-file config/acmpc.yaml

# T6  residual predictor (pushes one frame per controller tick; ready after 64)
source ~/acmpc_env.sh && ros2 run rdp_estimator estimator_node --ros-args --params-file config/rdp.yaml

# T7  logger -> runs/experiment_0/states.csv (one row per controller tick)
source ~/acmpc_env.sh && ros2 run state_logger logger_node --ros-args -p run_dir:=runs/experiment_0
```

Controller options (`-p name:=value` or edit `config/acmpc.yaml`):

| parameter | values |
|---|---|
| `mode` | `nmpc1` (default: hand-tuned NMPC, no checkpoint needed), `pid`, `acmpc` (the learned cost map; needs `artifacts/acmpc_adaptive/variants/C.pkl` from Notebook 5 or `artifacts/acmpc/model.pkl` from Notebook 3) |
| `level` | `gentle`, `moderate`, `aggressive` (Lissajous (9.5)), `hold` (fixed point `(0, 0, z0)`) |
| `horizon`, `n_iter` | MPC horizon and iLQR iterations. Default 5, 5 (~10 ms solve), flight-verified. Raise them only if the logged `loop_ms` stays well under 20 ms: 10/10 took ~23 ms under SITL load on 4 cores, and the vehicle flipped |
| `delay_steps` | latency compensation in control periods (default 2). The NMPC is tuned with zero delay and **diverges with one uncompensated 20 ms step** |
| `entry_s`, `entry_a_max` | hand-over transfer duration and acceleration cap (4 s, 3 m/s²) |
| `use_d`, `dmod_mode` | feed the RDP estimate into the prediction model, and how (§4.4) |

**Hand control to the controller** (in `pxh>`):

```
pxh> commander mode offboard
```

**Terminal 8: start the disturbance episode** (immediately after the switch)

```bash
source ~/acmpc_env.sh && ros2 run disturbance_manager manager_node --ros-args \
     --params-file config/disturbances.yaml -p scenario:=S2
```

It waits for the bridge of terminal 3, then runs the episode: 0–5 s nominal,
5–12 s scenario active (applied to `x500_0` in Gazebo and published as ground
truth), then recovery. Scenarios S0–S6 are listed in `config/disturbances.yaml`.
To stop: `pxh> commander land`, then `Ctrl-C` every terminal (the manager clears
its Gazebo wrench on exit), then `shutdown` in `pxh>`.

**Plots of a finished run**

```bash
source ~/acmpc_env.sh
ros2 run visualization live_panel runs/experiment_0     # -> runs/experiment_0/plots/R-F1, R-F2, R-F8
```

**Topics** (all ENU world / FLU body inside the workspace; PX4's NED/FRD is
converted at the controller boundary):

| topic | type | from → to |
|---|---|---|
| `/fmu/out/vehicle_odometry` | px4_msgs/VehicleOdometry | PX4 → controller, manager |
| `/fmu/out/actuator_motors` | px4_msgs/ActuatorMotors | PX4 → estimator, manager, logger |
| `/fmu/out/vehicle_status_v1` | px4_msgs/VehicleStatus | PX4 → controller (OFFBOARD starts the clock) |
| `/fmu/in/vehicle_rates_setpoint`, `/fmu/in/offboard_control_mode` | px4_msgs | controller → PX4 |
| `/acmpc/status` | std_msgs/Float64MultiArray (layout in `status_msg.py`) | controller → estimator, logger |
| `/rdp/disturbance_estimate` | geometry_msgs/WrenchStamped [N world, N·m body] | estimator → controller, logger |
| `/rdp/status` | std_msgs/Float64MultiArray [ms, fault, ready] | estimator → logger |
| `/disturbance/ground_truth`, `/disturbance/phase` | WrenchStamped, String | manager → logger (**never** the controller) |
| `/world/default/wrench/persistent`, `.../clear` | ros_gz_interfaces | manager → bridge → Gazebo |

### 5.9 Status of the deployment workspace: read before quoting results

**Flown in PX4 v1.16.2 SITL + Gazebo (gz-sim 8)** by following §5.8 step by
step. The run was headless on a 4-core machine; PX4 was built against conda-forge
Gazebo libraries because the OSRF apt repository was unreachable there, but the
PX4 source, the §5.1 DDS patch, the parameters and the ROS side were exactly as
written above:

| check | result |
|---|---|
| `check_ctbr --moment-scenario` (live) | checks 1, 2, 4 PASS; `actuator_motors` at 98 Hz |
| control loop | 50 Hz held: tick period 20.1 ms, solve 10.5 ms, complete loop 17.3 ms |
| hand-over | entry from the hover point, no transient, reference clock starts on OFFBOARD |
| S0, moderate Lissajous, 27 s | position error mean 6–19 cm (max 26 cm), no failsafe |
| S2 (2.43 N + 0.14 N·m step in Gazebo) | error 13 cm before, 25 cm mean / 46 cm max during, 17 cm after; no failsafe |

Three problems were found and fixed only by flying it, and you should know them:
- A direct hand-over to the path made the NMPC command zero thrust, and the
  vehicle flipped. The fix is the entry trajectory.
- One uncompensated control period of latency destabilises the NMPC. The fix is
  `delay_steps` (on the study plant: diverged → 9 cm max error at 1 tick of delay).
- A solve that overruns 20 ms under load also flips the vehicle. The fix is the
  lighter defaults.

Also verified ([docs/ROS2_WORKSPACE.md §7](docs/ROS2_WORKSPACE.md)):
`study/tests/test_ros_wiring.py` (19 tests) pins the nodes to the study. The
controller's reference, `u_ref`, AC-MPC observation and delay prediction equal
the study's `ref_state`, `Env.obs()` and `step_c`. The estimator frame equals
`_frame26`, PX4 odometry converts correctly, and the S1/S5 wrenches are
physically right. The disturbance path in gz-sim reproduces a step force
exactly (0.00 %) and a 0.5 Hz sinusoid to −0.7 %.

Limitations you need to know:

- **RDP accuracy depends on the exported model.** The committed `.npz` files come
  from a `smoke` training run (§3.2), fitted on position hold. In the SITL S2
  flight the estimate rose during the force step (to ~1.8 N of 2.43 N), but it
  was dominated by an oscillation at the Lissajous period. Retrain at `full`
  scale and re-export (§3.3) before reading anything into it.
- **S1 is quasi-static.** The payload's gravity and CG moment are applied; its
  inertial force and `inertia_scale` are not realised in Gazebo.
- `mode: acmpc` needs a trained checkpoint (§3.2, Notebooks 3/5). Without one,
  fly `nmpc1` (the default).
- Tilt reaches 30–50° on the moderate Lissajous with these weights. Start with
  `level:=gentle` or `level:=hold` on new hardware.

### 5.10 Without ROS

The algorithmic cores run without ROS, from the study venv:

```bash
cd $REPO && source .venv-study/bin/activate
PYTHONPATH="rdp_acmpc_ws/src/acmpc_controller:rdp_acmpc_ws/src/rdp_estimator:study" python -m acmpc_controller.check_ctbr
PYTHONPATH="rdp_acmpc_ws/src/acmpc_controller:rdp_acmpc_ws/src/rdp_estimator:study" python -m acmpc_controller.check_glue
PYTHONPATH="rdp_acmpc_ws/src/experiment_manager:study" python -m experiment_manager.runner
```

---

## 6. Configuration files

| file | content |
|---|---|
| `rdp_acmpc_ws/config/acmpc.yaml` | controller mode (`acmpc`/`nmpc1`/`pid`), horizon, iLQR iterations, reference A/B/ω/z₀, vehicle constants (asserted against the study by `check_glue`) |
| `rdp_acmpc_ws/config/rdp.yaml` | RDP model path, window H = 64, smoothing, watchdog budget |
| `rdp_acmpc_ws/config/disturbances.yaml` | scenario parameters S0–S6, timeline (on at 5 s, off at 12 s) |
| `rdp_acmpc_ws/config/experiments.yaml` | experiment matrix E-0, E-A … E-D and metric thresholds |
| `study/study_prelude.py` | `X500_SCALE` table (`smoke`/`medium`/`full`) |
| `lltc/lltc_hover.py` (top block) | `N_MPC`, sampling time, weights, bounds, sampling box, penalties |

---

## 7. Documentation index

| document | read it for |
|---|---|
| [lltc/README.md](lltc/README.md) | the LLTC hover study: method, settings, results |
| [docs/ROS2_WORKSPACE.md](docs/ROS2_WORKSPACE.md) | architecture of the ROS 2 workspace, data flow, each package, check binaries |
| [docs/CORRECTIONS.md](docs/CORRECTIONS.md) | the 8 errors found in the build specification, each with its measurement and test |
| [docs/AUDIT.md](docs/AUDIT.md) | audit against the real PX4 x500 (SDF + airframe) and the two source papers |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | JAX/XLA `tile size` crash, verified versions |

Key facts these documents establish, so that you do not have to rediscover them:

- PX4's `SIM_GZ_EC_MIN = 150` idle floor gives `u_hover = 0.7287`, not 0.7694.
  Ignoring it makes hover control effectiveness 17.65 % optimistic.
- The allocator's `CA_ROTORn_KM = 0.05` differs from the rotors' 0.016. This is
  a real 3.125× yaw-authority mismatch, and it is modelled.
- Ground truth never reaches the controller. The RDP window is fed only from
  state and actuator topics.
- Latency is quoted only as `solve_latency_ms` on a batch of one, against the
  20 ms period.

## 8. References

- S. Abdufattokhov, M. Zanon, A. Bemporad, *Learning Lyapunov terminal costs from
  data for complexity reduction in nonlinear MPC*, Int. J. Robust Nonlinear
  Control 34(13), 2024. doi:10.1002/rnc.7411
- S. Gros, M. Zanon, *Data-driven economic NMPC using reinforcement learning*,
  IEEE TAC 2020, arXiv:1904.04152. This is the licence to learn the cost.
- A. Romero, Y. Song, D. Scaramuzza, *Actor-Critic Model Predictive Control*,
  ICRA 2024, arXiv:2306.09852. This is the cost-map-over-short-MPC architecture.
- C. Büskens, H. Maurer, *Sensitivity analysis and real-time optimization of
  parametric nonlinear programming problems*, 2001.
- W. Li, E. Todorov (2004) and Y. Tassa et al. (2012): iLQR and its
  regularisation.
- PX4 ROS 2 user guide: https://docs.px4.io/main/en/ros2/user_guide
