#!/usr/bin/env bash
set -euo pipefail

# Small process supervisor for the remote host.  It deliberately manages one
# configured campaign and its own Controller launcher only; use systemd for a
# fleet, not a broad process matcher here.
ACTION="${1:-}"
if [ -z "$ACTION" ] || [[ "$ACTION" != "start" && "$ACTION" != "status" && "$ACTION" != "stop" && "$ACTION" != "pause" && "$ACTION" != "resume" && "$ACTION" != "ensure-controller" && "$ACTION" != "cleanup-controller" ]]; then
    echo "usage: $0 start|status|stop|pause|resume" >&2
    exit 2
fi

REPO_ROOT="${REMOTE_CAMPAIGN_REPO_ROOT:-/home/intern/huangjiahao/Harness4H3-rsi}"
# The remote H3 benchmark/evaluator needs the installed CUDA/PyTorch/OpenCV
# environment.  Callers may still override this for a CPU-only campaign.
PYTHON="${REMOTE_CAMPAIGN_PYTHON:-/home/intern/miniconda3/envs/comfy/bin/python}"
CONFIG="${REMOTE_CAMPAIGN_CONFIG:-$REPO_ROOT/configs/remote-l40-h3-rsi-overnight.yaml}"
OUTPUT="${REMOTE_CAMPAIGN_OUTPUT:-$REPO_ROOT/var/remote-h3-controller-20260914}"
STATE_FILE="$OUTPUT/campaign_state.json"
PID_FILE="${REMOTE_CAMPAIGN_PID_FILE:-$OUTPUT/.remote-campaign.pid}"
LOG_FILE="${REMOTE_CAMPAIGN_LOG_FILE:-$OUTPUT/remote-campaign.log}"
LOCK_FILE="$OUTPUT/.overnight-controller.lock"
STOP_FILE="${REMOTE_CAMPAIGN_STOP_FILE:-$OUTPUT/.stop-requested}"
PAUSE_FILE="${REMOTE_CAMPAIGN_PAUSE_FILE:-$OUTPUT/.operator-paused}"
HANDOFF_HOLD_FILE="${REMOTE_CAMPAIGN_HANDOFF_HOLD_FILE:-$REPO_ROOT/work/remote-h3-controller-20260914/.controller-handoff-hold.json}"
CAMPAIGN_ROOT="$(dirname -- "$HANDOFF_HOLD_FILE")"
RUNNER="$REPO_ROOT/tools/run_overnight_controller.py"
SUPERVISOR="$REPO_ROOT/tools/remote-campaign-supervisor.sh"
CONTROLLER_LAUNCHER="${REMOTE_CONTROLLER_LAUNCHER:-$REPO_ROOT/tools/controller-wait-launch.sh}"
MANAGE_CONTROLLER="${REMOTE_CAMPAIGN_MANAGE_CONTROLLER:-1}"
CONTROLLER_LAUNCHER_PID_FILE="${REMOTE_CONTROLLER_LAUNCHER_PID_FILE:-$CAMPAIGN_ROOT/.controller-launcher.pid}"
CONTROLLER_LAUNCHER_LOG="${REMOTE_CONTROLLER_LAUNCHER_LOG:-$CAMPAIGN_ROOT/controller-launcher.log}"
CONTROLLER_LAUNCHER_STOP_GRACE_SECONDS="${REMOTE_CONTROLLER_LAUNCHER_STOP_GRACE_SECONDS:-30}"
# The configured remote Controller is normally exposed on 8001, with the
# legacy local endpoint on 8000 as a fallback.  Keep these defaults in the
# supervisor so an unattended restart does not silently probe the wrong port.
CONTROLLER_REMOTE_PORT="${REMOTE_CAMPAIGN_CONTROLLER_PORT:-8001}"
CONTROLLER_FALLBACK_PORTS="${REMOTE_CAMPAIGN_CONTROLLER_FALLBACK_PORTS:-8000}"
# A released evaluator GPU should be reused quickly, while the detached
# campaign remains cheap during a long shared-host wait.  Fifteen seconds is
# short enough to avoid a visible post-Geneval gap without busy-spinning.
RESOURCE_POLL_INTERVAL_S="${REMOTE_CAMPAIGN_RESOURCE_POLL_INTERVAL_S:-15}"
CAMPAIGN_MAX_ITERATIONS="${REMOTE_CAMPAIGN_MAX_ITERATIONS:-512}"

read_pid() {
    [ -f "$PID_FILE" ] || return 1
    local value
    value=$(tr -d '[:space:]' < "$PID_FILE")
    case "$value" in
        ''|*[!0-9]*) return 1 ;;
        *) printf '%s\n' "$value" ;;
    esac
}

pid_is_campaign() {
    local pid="$1" command
    command=$(ps -p "$pid" -o args= 2>/dev/null || true)
    [[ "$command" == *"run_overnight_controller.py"* || "$command" == *"remote-campaign-supervisor.sh"* ]]
}

pid_is_controller_launcher() {
    local pid="$1" command
    command=$(ps -p "$pid" -o args= 2>/dev/null || true)
    [[ "$command" == *"$CONTROLLER_LAUNCHER"* ]]
}

read_controller_launcher_pid() {
    [ -f "$CONTROLLER_LAUNCHER_PID_FILE" ] || return 1
    local value
    value=$(tr -d '[:space:]' < "$CONTROLLER_LAUNCHER_PID_FILE")
    case "$value" in
        ''|*[!0-9]*) return 1 ;;
        *) printf '%s\n' "$value" ;;
    esac
}

find_controller_launcher_pid() {
    local pid command
    while read -r pid command; do
        [ -n "$pid" ] || continue
        if [[ "$command" == *"$CONTROLLER_LAUNCHER"* ]] && [ "$pid" != "$$" ]; then
            printf '%s\n' "$pid"
            return 0
        fi
    done < <(ps -eo pid=,args= 2>/dev/null || true)
    return 1
}

controller_launcher_pid() {
    local pid
    pid=$(read_controller_launcher_pid || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && pid_is_controller_launcher "$pid"; then
        printf '%s\n' "$pid"
        return 0
    fi
    pid=$(find_controller_launcher_pid || true)
    if [ -n "$pid" ]; then
        printf '%s\n' "$pid"
        return 0
    fi
    return 1
}

ensure_controller_launcher() {
    [ "$MANAGE_CONTROLLER" != "0" ] || return 0
    [ -f "$CONTROLLER_LAUNCHER" ] || {
        echo "Controller launcher not found: $CONTROLLER_LAUNCHER" >&2
        return 1
    }
    mkdir -p -- "$CAMPAIGN_ROOT" "$(dirname -- "$CONTROLLER_LAUNCHER_LOG")"
    local pid
    pid=$(controller_launcher_pid || true)
    if [ -n "$pid" ]; then
        printf '%s\n' "$pid" > "$CONTROLLER_LAUNCHER_PID_FILE"
        echo "Controller launcher already running pid=$pid"
        return 0
    fi
    nohup env \
        CONTROLLER_GPU_LEASE_FILE="$CAMPAIGN_ROOT/.controller-gpu-lease.json" \
        CONTROLLER_HANDOFF_HOLD_FILE="$CAMPAIGN_ROOT/.controller-handoff-hold.json" \
        CONTROLLER_WORKER_LEASE_FILE="$CAMPAIGN_ROOT/.h3-worker-gpu-lease.json" \
        CONTROLLER_COMFY_LEASE_FILE="$CAMPAIGN_ROOT/.comfyui-gpu-lease.json" \
        bash "$CONTROLLER_LAUNCHER" \
        >> "$CONTROLLER_LAUNCHER_LOG" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$CONTROLLER_LAUNCHER_PID_FILE"
    echo "started Controller launcher pid=$pid log=$CONTROLLER_LAUNCHER_LOG"
}

stop_controller_launcher() {
    [ "$MANAGE_CONTROLLER" != "0" ] || return 0
    local pid
    pid=$(read_controller_launcher_pid || true)
    if [ -z "$pid" ]; then
        rm -f -- "$CONTROLLER_LAUNCHER_PID_FILE"
        return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        rm -f -- "$CONTROLLER_LAUNCHER_PID_FILE"
        return 0
    fi
    if ! pid_is_controller_launcher "$pid"; then
        echo "refusing to stop PID $pid: command is not the configured Controller launcher" >&2
        return 1
    fi
    kill_process_tree "$pid" TERM
    local deadline=$((SECONDS + CONTROLLER_LAUNCHER_STOP_GRACE_SECONDS))
    while kill -0 "$pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill_process_tree "$pid" KILL
    fi
    rm -f -- "$CONTROLLER_LAUNCHER_PID_FILE"
    echo "stopped Controller launcher pid=$pid"
}

campaign_is_live() {
    local pid
    pid=$(read_pid || true)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && pid_is_campaign "$pid"
}

campaign_has_finished() {
    ! campaign_is_live && ! lock_is_held && [ -f "$OUTPUT/overnight-result.json" ]
}

lock_is_held() {
    command -v flock >/dev/null 2>&1 || return 1
    ! flock -n "$LOCK_FILE" -c true 2>/dev/null
}

clear_dead_handoff_hold() {
    # A graceful campaign stop can happen between creating the handoff hold
    # and the normal finally-based cleanup.  Do not leave the Controller
    # watcher paused until its long TTL expires: only remove a hold whose
    # recorded campaign owner is no longer alive, and never touch a live
    # campaign's reservation.
    [ -f "$HANDOFF_HOLD_FILE" ] || return 0
    if "$PYTHON" - "$HANDOFF_HOLD_FILE" <<'PY'
import json
import os
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    owner_pid = int(payload.get("owner_pid", 0))
except (OSError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(0)

if owner_pid <= 0:
    raise SystemExit(0)
try:
    with open(f"/proc/{owner_pid}/stat", "r", encoding="utf-8") as handle:
        fields = handle.read().split()
    # A zombie is no longer an active campaign owner even though kill(pid, 0)
    # still succeeds on Linux.
    raise SystemExit(0 if len(fields) < 3 or fields[2] != "Z" else 1)
except FileNotFoundError:
    raise SystemExit(1)
except OSError:
    raise SystemExit(0)
PY
    then
        return 0
    fi
    rm -f -- "$HANDOFF_HOLD_FILE"
    echo "removed stale Controller handoff hold: $HANDOFF_HOLD_FILE"
}

kill_process_tree() {
    local root="$1" signal="$2" child
    for child in $(pgrep -P "$root" 2>/dev/null || true); do
        kill_process_tree "$child" "$signal"
    done
    kill -"$signal" "$root" 2>/dev/null || true
}

status() {
    local pid="" state="missing"
    pid=$(read_pid || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && pid_is_campaign "$pid"; then
        if [ -f "$STOP_FILE" ]; then
            state="stop_requested"
        else
            state="running"
        fi
    elif [ -f "$PAUSE_FILE" ]; then
        state="paused"
    elif [ -n "$pid" ]; then
        state="stale_pid_file"
    elif lock_is_held; then
        state="running_without_service_pid"
    fi
    printf 'campaign=%s pid=%s state=%s\n' "$OUTPUT" "${pid:-none}" "$state"
    if [ -f "$STATE_FILE" ]; then
        "$PYTHON" - "$STATE_FILE" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        state = json.load(handle)
    pipeline = state.get("pipeline", {}) if isinstance(state, dict) else {}
    pending = state.get("pending_plan", {}) if isinstance(state, dict) else {}
    recovery = state.get("resource_replan_intent") if isinstance(state, dict) else None
    print(json.dumps({
        "current_model_id": state.get("current_model_id"),
        "training_calls": state.get("training_calls"),
        "pending_plan_experiment_id": pending.get("experiment_id") if isinstance(pending, dict) else None,
        "pending_attempts": state.get("pending_attempts"),
        "resource_replan_intent": recovery,
        "pipeline_stage": pipeline.get("stage"),
        "pipeline_iteration": pipeline.get("iteration"),
        "pipeline_waiting_on": state.get("pipeline_waiting_on"),
        "last_resource_decision": state.get("last_resource_decision"),
    }, ensure_ascii=False, sort_keys=True))
except Exception as exc:
    print(json.dumps({"state_error": str(exc)[:500]}, ensure_ascii=False))
PY
    else
        echo 'campaign_state=missing'
    fi
}

if [ "$ACTION" = "ensure-controller" ]; then
    ensure_controller_launcher
    exit 0
fi

if [ "$ACTION" = "cleanup-controller" ]; then
    stop_controller_launcher
    exit 0
fi

if [ "$ACTION" = "resume" ]; then
    rm -f -- "$PAUSE_FILE"
    echo "campaign automation resumed; no campaign was started"
    exit 0
fi

if [ "$ACTION" = "pause" ]; then
    mkdir -p -- "$OUTPUT"
    touch -- "$PAUSE_FILE"
    pid="$(read_pid || true)"
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
        rm -f -- "$PID_FILE"
        pid=""
    fi
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && pid_is_campaign "$pid"; then
        # Pause is deliberately graceful: it creates the same boundary marker
        # as stop and leaves checkpoint/experience finalization to the runner.
        touch -- "$STOP_FILE"
        echo "campaign automation paused; graceful stop requested pid=$pid marker=$STOP_FILE"
    else
        echo "campaign automation paused; no campaign is running"
    fi
    exit 0
fi

case "$ACTION" in
    status)
        status
        ;;
    start)
        mkdir -p -- "$OUTPUT"
        if [ -f "$PAUSE_FILE" ]; then
            echo "campaign automation is operator-paused: $PAUSE_FILE" >&2
            exit 3
        fi
        if lock_is_held; then
            echo "campaign lock is held; refusing a second launch: $LOCK_FILE" >&2
            exit 1
        fi
        existing="$(read_pid || true)"
        if [ -n "$existing" ] && kill -0 "$existing" 2>/dev/null && pid_is_campaign "$existing"; then
            echo "already running pid=$existing"
            exit 0
        fi
        if [ -n "$existing" ]; then
            rm -f -- "$PID_FILE"
        fi
        # A previous graceful stop is a one-shot request.  Do not make a
        # later explicit start exit immediately because the marker survived.
        rm -f -- "$STOP_FILE"
        clear_dead_handoff_hold
        if [ ! -f "$RUNNER" ]; then
            echo "runner not found: $RUNNER" >&2
            exit 1
        fi
        if [ ! -f "$SUPERVISOR" ]; then
            echo "campaign supervisor not found: $SUPERVISOR" >&2
            exit 1
        fi
        ensure_controller_launcher
        nohup env \
            REMOTE_CAMPAIGN_REPO_ROOT="$REPO_ROOT" \
            REMOTE_CAMPAIGN_PYTHON="$PYTHON" \
            REMOTE_CAMPAIGN_CONFIG="$CONFIG" \
            REMOTE_CAMPAIGN_OUTPUT="$OUTPUT" \
            REMOTE_CAMPAIGN_STOP_FILE="$STOP_FILE" \
            REMOTE_CAMPAIGN_CONTROLLER_PORT="$CONTROLLER_REMOTE_PORT" \
            REMOTE_CAMPAIGN_CONTROLLER_FALLBACK_PORTS="$CONTROLLER_FALLBACK_PORTS" \
            REMOTE_CAMPAIGN_RESOURCE_POLL_INTERVAL_S="$RESOURCE_POLL_INTERVAL_S" \
            REMOTE_CAMPAIGN_MAX_ITERATIONS="$CAMPAIGN_MAX_ITERATIONS" \
            bash "$SUPERVISOR" \
            >> "$LOG_FILE" 2>&1 < /dev/null &
        pid=$!
        printf '%s\n' "$pid" > "$PID_FILE"
        echo "started pid=$pid log=$LOG_FILE"
        ;;
    stop)
        pid="$(read_pid || true)"
        if [ -z "$pid" ]; then
            if campaign_has_finished; then
                stop_controller_launcher
                echo "campaign not running; owned Controller launcher cleaned up"
            else
                echo "not running"
            fi
            exit 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f -- "$PID_FILE"
            if campaign_has_finished; then
                stop_controller_launcher
                echo "stale pid file removed; owned Controller launcher cleaned up"
            else
                echo "stale pid file removed"
            fi
            exit 0
        fi
        if ! pid_is_campaign "$pid"; then
            echo "refusing to stop PID $pid: command is not run_overnight_controller.py" >&2
            exit 1
        fi
        # The campaign loop checks this exact marker at iteration boundaries
        # and waits for a speculative successor to publish its durable result.
        # Returning immediately keeps a service handoff from interrupting a
        # multi-GPU checkpoint write; the next status call reports the
        # requested state while the supervisor drains safely.
        touch -- "$STOP_FILE"
        echo "graceful stop requested pid=$pid marker=$STOP_FILE"
        ;;
esac
