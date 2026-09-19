#!/usr/bin/env bash
# Fly the S7 payload demonstration: three controllers, one hover hold each,
# one accumulating Z-X figure.
#
#   ./tools/run_hover_payload_demo.sh                 # all three, 40 s each
#   ./tools/run_hover_payload_demo.sh --duration 60
#   ./tools/run_hover_payload_demo.sh --only acmpc_adaptive
#   ./tools/run_hover_payload_demo.sh --headless      # no window, PNG only
#
# It does NOT start PX4, Gazebo or the uXRCE-DDS agent -- those are yours to
# start and to watch.  It refuses to fly if the preflight fails, because every
# symptom of a PX4 version mismatch downstream of here is silent.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"

CONTROLLERS=(nmpc1 acmpc acmpc_adaptive)
DURATION=40
# PX4's offboard failsafe lands the vehicle when the setpoint stream stops, so
# each leg ends on the ground and the next takes off from the same state.  A
# landing from 1.5 m plus the disarm takes longer than it looks; 20 s is not
# padding, it is what makes the three legs share a starting condition.
SETTLE=20
SESSION="runs/demo_s7_$(date +%Y%m%d_%H%M%S)"
HEADLESS=false
NAMESPACE=""
SKIP_PREFLIGHT=false
HOLD=(0.0 0.0 1.5)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --settle)   SETTLE="$2"; shift 2 ;;
    --only)     CONTROLLERS=("$2"); shift 2 ;;
    --session)  SESSION="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --hold)     HOLD=("$2" "$3" "$4"); shift 4 ;;
    --headless) HEADLESS=true; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=true; shift ;;
    -h|--help)  sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# The payload is defined ONCE, in disturbance_manager.scenarios.  Read it here
# rather than writing the numbers a fourth time: check_glue can see a drift
# between the SDF patch, the scenario and the plotter, but it cannot see one in
# a shell literal.
read -r PM PRX PRY PRZ < <(PYTHONPATH=src/disturbance_manager python3 -c \
  "from disturbance_manager.scenarios import S7_PAYLOAD_M as m, S7_PAYLOAD_R as r; print(m, *r)")

mkdir -p "$SESSION"
echo "=============================================================="
echo " S7 hover hold under an asymmetric payload"
echo " session     $SESSION"
echo " controllers ${CONTROLLERS[*]}"
echo " duration    ${DURATION} s each"
echo " hold        (${HOLD[*]})"
echo "=============================================================="

# ---------------------------------------------------------------- preflight --
# The payload must be on the MODEL.  Publishing the truth without attaching the
# mass would give three identical flights and three confident, wrong plots.
echo
echo "--- Gazebo payload ---"
python3 tools/payload_sdf.py --status || true
if ! python3 tools/payload_sdf.py --status 2>/dev/null | grep -q INSTALLED; then
  cat >&2 <<'MSG'

  The payload is NOT attached to the Gazebo model.  Without it every controller
  flies an unloaded vehicle and the comparison measures nothing.  Attach it and
  RESTART SITL (Gazebo caches models):

      python3 tools/payload_sdf.py --install

MSG
  exit 1
fi

if [[ "$SKIP_PREFLIGHT" != true ]]; then
  echo
  echo "--- PX4 topics and message versions ---"
  NS_ARG=()
  [[ -n "$NAMESPACE" ]] && NS_ARG=(--namespace "$NAMESPACE")
  if ! ros2 run acmpc_controller check_px4 "${NS_ARG[@]}"; then
    echo "preflight failed; refusing to fly.  --skip-preflight overrides." >&2
    exit 1
  fi
fi

# ------------------------------------------------------------------- flights --
for C in "${CONTROLLERS[@]}"; do
  echo
  echo "=============================================================="
  echo " flying $C for ${DURATION} s"
  echo "=============================================================="
  LOG="$SESSION/${C}.log"
  set +e
  timeout --signal=INT --kill-after=10 "$DURATION" \
    ros2 launch acmpc_controller hover_payload_demo.launch.py \
      controller:="$C" \
      session_dir:="$SESSION" \
      hold_x:="${HOLD[0]}" hold_y:="${HOLD[1]}" hold_z:="${HOLD[2]}" \
      headless:="$HEADLESS" \
      px4_namespace:="$NAMESPACE" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  set -e
  # 124 is timeout's "I stopped it", which is the expected end of a leg
  if [[ $RC -ne 0 && $RC -ne 124 && $RC -ne 130 ]]; then
    echo "leg $C exited with $RC; see $LOG" >&2
  fi
  if grep -qi "VERSION MISMATCH\|no topic matching" "$LOG"; then
    echo >&2
    echo "PX4 topic resolution failed during $C -- see $LOG.  Stopping: the" >&2
    echo "remaining legs would fail the same way." >&2
    exit 1
  fi
  echo "--- $C done; waiting ${SETTLE}s for the failsafe land + disarm ---"
  sleep "$SETTLE"
done

# -------------------------------------------------------------------- figure --
echo
echo "=============================================================="
echo " rendering the comparison"
echo "=============================================================="
ros2 run visualization live_zx --replay "$SESSION" \
  --payload "$PM" "$PRX" "$PRY" "$PRZ" \
  --setpoint "${HOLD[0]}" "${HOLD[1]}" "${HOLD[2]}"

echo
echo "session   $SESSION"
echo "figure    $SESSION/R-F13_zx_demo.png"
echo "traces    $SESSION/zx_traces.npz"
echo "metrics   $SESSION/zx_metrics.json"
echo
echo "To take the payload back off the Gazebo model:"
echo "    python3 tools/payload_sdf.py --revert"
