#!/usr/bin/env bash
# Fly the S7 payload demonstration: a hover hold per controller, accumulating
# into one Z-X figure.
#
#   ./tools/run_hover_payload_demo.sh                       # all three, 40 s each
#   ./tools/run_hover_payload_demo.sh acmpc_adaptive        # just this one
#   ./tools/run_hover_payload_demo.sh nmpc1 acmpc           # these two, in order
#   ./tools/run_hover_payload_demo.sh acmpc --duration 60
#   ./tools/run_hover_payload_demo.sh --headless            # no window, PNG only
#
# Controllers: nmpc1 | acmpc | acmpc_adaptive | pid
#
# Naming a controller REUSES the newest session directory rather than starting
# an empty one, so flying them one at a time accumulates into the same figure.
# --session forces a particular one; --new-session forces a fresh one.
#
# It does NOT start PX4, Gazebo or the uXRCE-DDS agent -- those are yours to
# start and to watch.  It refuses to fly if the preflight fails, because every
# symptom of a PX4 version mismatch downstream of here is silent.
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WS"

KNOWN=(nmpc1 acmpc acmpc_adaptive pid)
CONTROLLERS=()
DURATION=40
# PX4's offboard failsafe lands the vehicle when the setpoint stream stops, so
# each leg ends on the ground and the next takes off from the same state.  A
# landing from 1.5 m plus the disarm takes longer than it looks; 20 s is not
# padding, it is what makes the three legs share a starting condition.
SETTLE=20
SESSION=""
NEW_SESSION=false
HEADLESS=false
NAMESPACE=""
SKIP_PREFLIGHT=false
HOLD=(0.0 0.0 1.5)

is_known() { local c; for c in "${KNOWN[@]}"; do [[ "$c" == "$1" ]] && return 0; done; return 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    --settle)   SETTLE="$2"; shift 2 ;;
    --only)     CONTROLLERS+=("$2"); shift 2 ;;   # kept; same as naming it
    --session)  SESSION="$2"; shift 2 ;;
    --new-session) NEW_SESSION=true; shift ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --hold)     HOLD=("$2" "$3" "$4"); shift 4 ;;
    --headless) HEADLESS=true; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=true; shift ;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *)  # a bare word is a controller name
      if is_known "$1"; then
        CONTROLLERS+=("$1"); shift
      else
        echo "unknown controller: $1" >&2
        echo "expected one of: ${KNOWN[*]}" >&2
        exit 2
      fi ;;
  esac
done

#  No controller named -> fly the comparison, in the order the report reads it.
PICKED=true
if [[ ${#CONTROLLERS[@]} -eq 0 ]]; then
  CONTROLLERS=(nmpc1 acmpc acmpc_adaptive)
  PICKED=false
fi

#  Flying one leg at a time should ACCUMULATE into the figure, not start a new
#  and empty one each time.  So when a controller was named explicitly, reuse
#  the newest session unless told otherwise.
if [[ -z "$SESSION" ]]; then
  if [[ "$PICKED" == true && "$NEW_SESSION" != true ]]; then
    SESSION="$(ls -1d runs/demo_s7_* 2>/dev/null | sort | tail -1 || true)"
  fi
  if [[ -z "$SESSION" ]]; then
    SESSION="runs/demo_s7_$(date +%Y%m%d_%H%M%S)"
  else
    echo "reusing session $SESSION  (--new-session to start a fresh one)"
  fi
fi

# The payload is defined ONCE, in disturbance_manager.scenarios.  Read it here
# rather than writing the numbers a fourth time: check_glue can see a drift
# between the SDF patch, the scenario and the plotter, but it cannot see one in
# a shell literal.
read -r PM PRX PRY PRZ < <(PYTHONPATH=src/disturbance_manager python3 -c \
  "from disturbance_manager.scenarios import S7_PAYLOAD_M as m, S7_PAYLOAD_R as r; print(m, *r)")

mkdir -p "$SESSION"

#  How many samples the session's trace file holds for one controller.  Prints
#  0 when the file, the key or numpy is missing -- the caller only compares it
#  against itself, so "cannot tell" and "nothing there" are the same answer.
trace_samples() {
  python3 - "$SESSION/zx_traces.npz" "$1" <<'PY' 2>/dev/null || echo 0
import sys
try:
    import numpy as np
    z = np.load(sys.argv[1], allow_pickle=False)
    print(int(z[f"{sys.argv[2]}/t"].size) if f"{sys.argv[2]}/t" in z.files else 0)
except Exception:
    print(0)
PY
}

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
#  Capture ONCE and test the string.  Piping this into `grep -q` looks tidier
#  and is wrong: grep exits at the first match, python takes SIGPIPE, and
#  `set -o pipefail` then reports the pipeline as failed -- so an installed
#  payload intermittently reads as missing, depending on whether python
#  finished writing before grep stopped reading.
PAYLOAD_STATUS="$(python3 tools/payload_sdf.py --status 2>&1 || true)"
echo "$PAYLOAD_STATUS"
if ! grep -q INSTALLED <<<"$PAYLOAD_STATUS"; then
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

  #  `ros2 launch` rejects `name:=` with an empty value outright, so an unused
  #  optional argument must be OMITTED rather than passed empty.  Build the
  #  list instead of interpolating, which is also how a future optional
  #  argument gets added without repeating this bug.
  LAUNCH_ARGS=(
    controller:="$C"
    session_dir:="$SESSION"
    hold_x:="${HOLD[0]}" hold_y:="${HOLD[1]}" hold_z:="${HOLD[2]}"
    headless:="$HEADLESS"
  )
  [[ -n "$NAMESPACE" ]] && LAUNCH_ARGS+=(px4_namespace:="$NAMESPACE")

  BEFORE=$(trace_samples "$C")
  START=$SECONDS
  set +e
  timeout --signal=INT --kill-after=10 "$DURATION" \
    ros2 launch acmpc_controller hover_payload_demo.launch.py \
      "${LAUNCH_ARGS[@]}" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  set -e
  ELAPSED=$((SECONDS - START))

  #  124 is timeout saying "I stopped it", which is how a healthy leg ends.
  #  Anything else means the leg did not run to completion, and the remaining
  #  legs would fail the same way -- so stop here instead of burning
  #  ${SETTLE}s apiece to reach an empty figure and an opaque error.
  if [[ $RC -ne 124 && $RC -ne 130 ]]; then
    echo >&2
    echo "leg '$C' did not run to completion (exit $RC after ${ELAPSED}s)." >&2
    echo "The last lines of $LOG:" >&2
    tail -n 12 "$LOG" | sed 's/^/    /' >&2
    echo >&2
    echo "Stopping: the remaining legs would fail the same way." >&2
    exit 1
  fi
  if grep -qi "VERSION MISMATCH\|no topic matching" "$LOG"; then
    echo >&2
    echo "PX4 topic resolution failed during '$C' -- see $LOG.  Stopping." >&2
    exit 1
  fi

  #  A leg that launched, ran its ${DURATION}s and recorded NOTHING is the
  #  quiet failure this demo is most exposed to: the vehicle never armed, or
  #  the plotter never saw a pose.  Say so now, next to the log that explains
  #  it, rather than at the render step three minutes later.
  AFTER=$(trace_samples "$C")
  if [[ "$AFTER" -le "$BEFORE" ]]; then
    echo >&2
    echo "WARNING: leg '$C' recorded no new samples (${BEFORE} -> ${AFTER})." >&2
    echo "  The usual causes, in order of likelihood:" >&2
    echo "    * the vehicle never armed -- grep '$LOG' for 'offboard requested'" >&2
    echo "    * PX4/Gazebo is not running, or the uXRCE-DDS agent is not" >&2
    echo "    * the controller crashed on start -- see the tail above" >&2
  else
    echo "--- $C recorded $((AFTER - BEFORE)) samples ---"
  fi

  echo "--- $C done; waiting ${SETTLE}s for the failsafe land + disarm ---"
  sleep "$SETTLE"
done

# -------------------------------------------------------------------- figure --
echo
echo "=============================================================="
echo " rendering the comparison"
echo "=============================================================="
if [[ ! -f "$SESSION/zx_traces.npz" ]]; then
  echo "No traces were recorded in $SESSION, so there is nothing to render." >&2
  echo >&2
  echo "Every leg launched and exited cleanly, so the graph was up but the" >&2
  echo "plotter never saw a pose.  Check, in this order:" >&2
  echo "  1. ros2 topic hz /fmu/out/vehicle_odometry      -- is PX4 publishing?" >&2
  echo "  2. grep -l 'offboard requested' $SESSION/*.log  -- did it arm?" >&2
  echo "  3. tail -40 $SESSION/*.log                      -- did a node die?" >&2
  exit 1
fi

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
echo
echo "To add or re-fly one controller into this same session:"
echo "    ./tools/run_hover_payload_demo.sh <controller> --session $SESSION"
