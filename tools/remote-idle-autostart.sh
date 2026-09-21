#!/usr/bin/env bash
set -euo pipefail

# Remote-side idle gate. It is intentionally CPU-only while the shared host is
# busy, and starts the project service only after every configured GPU passes
# both the compute-process and free-memory checks.
ACTION="${1:-status}"
if [[ "$ACTION" != "start" && "$ACTION" != "status" && "$ACTION" != "stop" && "$ACTION" != "run" ]]; then
    echo "usage: $0 start|status|stop; pause/resume via remote-campaign-service.sh" >&2
    exit 2
fi

REPO_ROOT="${REMOTE_CAMPAIGN_REPO_ROOT:-/home/intern/huangjiahao/Harness4H3-rsi}"
SERVICE="$REPO_ROOT/tools/remote-campaign-service.sh"
WATCH_ROOT="${REMOTE_IDLE_AUTOSTART_ROOT:-$REPO_ROOT/work/remote-h3-controller-20260914}"
PID_FILE="${REMOTE_IDLE_AUTOSTART_PID_FILE:-$WATCH_ROOT/.idle-autostart.pid}"
STOP_FILE="${REMOTE_IDLE_AUTOSTART_STOP_FILE:-$WATCH_ROOT/.idle-autostart-stop}"
START_MARKER="${REMOTE_IDLE_AUTOSTART_START_MARKER:-$WATCH_ROOT/.idle-autostart-started-at}"
LOG_FILE="${REMOTE_IDLE_AUTOSTART_LOG_FILE:-$WATCH_ROOT/idle-autostart.log}"
OUTPUT="${REMOTE_CAMPAIGN_OUTPUT:-$REPO_ROOT/var/remote-h3-controller-20260914}"
PAUSE_FILE="${REMOTE_CAMPAIGN_PAUSE_FILE:-$OUTPUT/.operator-paused}"
POLL_SECONDS="${REMOTE_IDLE_AUTOSTART_POLL_SECONDS:-30}"
GPU_COUNT="${REMOTE_IDLE_AUTOSTART_GPU_COUNT:-4}"
MIN_FREE_MIB="${REMOTE_IDLE_AUTOSTART_MIN_FREE_MIB:-20000}"
IDLE_STABILITY_SAMPLES="${REMOTE_IDLE_AUTOSTART_STABILITY_SAMPLES:-3}"
IDLE_STABILITY_POLL_SECONDS="${REMOTE_IDLE_AUTOSTART_STABILITY_POLL_SECONDS:-5}"
COMFY_GPU_INDEX="${REMOTE_IDLE_AUTOSTART_COMFY_GPU_INDEX:-0}"
COMFY_PORT="${REMOTE_IDLE_AUTOSTART_COMFY_PORT:-8188}"
COMFY_MAX_IDLE_USED_MIB="${REMOTE_IDLE_AUTOSTART_COMFY_MAX_IDLE_USED_MIB:-2048}"

read_pid() {
    [ -f "$PID_FILE" ] || return 1
    local value
    value=$(tr -d '[:space:]' < "$PID_FILE")
    case "$value" in
        ''|*[!0-9]*) return 1 ;;
        *) printf '%s\n' "$value" ;;
    esac
}

pid_is_watcher() {
    local pid="$1" command
    command=$(ps -p "$pid" -o args= 2>/dev/null || true)
    [[ "$command" == *"remote-idle-autostart.sh run"* ]]
}

service_is_live() {
    local state
    state=$(bash "$SERVICE" status 2>/dev/null || true)
    [[ "$state" == *"state=running"* || "$state" == *"state=stop_requested"* || "$state" == *"state=running_without_service_pid"* ]]
}

terminal_result_is_newer_than_start() {
    [ -f "$OUTPUT/overnight-result.json" ] || return 1
    [ -f "$START_MARKER" ] || return 1
    local result_time start_time
    result_time=$(stat -c %Y -- "$OUTPUT/overnight-result.json" 2>/dev/null || stat -f %m -- "$OUTPUT/overnight-result.json" 2>/dev/null || true)
    start_time=$(tr -d '[:space:]' < "$START_MARKER" 2>/dev/null || true)
    case "$result_time:$start_time" in
        ''|*[!0-9:]*|*:*:*) return 1 ;;
    esac
    [ "$result_time" -ge "$start_time" ]
}

gpu_idle() {
    local rows count=0 index used total free apps pid process_name used_memory args queue
    rows=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null) || return 1
    while IFS=, read -r index used total; do
        index="${index//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        total="${total//[[:space:]]/}"
        case "$index:$used:$total" in
            ''|*[!0-9:]*|*:*:*:*) return 1 ;;
        esac
        free=$((total - used))
        [ "$free" -ge "$MIN_FREE_MIB" ] || return 1
        apps=$(nvidia-smi --id="$index" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null) || return 1
        while IFS=, read -r pid process_name used_memory; do
            pid="${pid//[[:space:]]/}"
            used_memory="${used_memory//[[:space:]]/}"
            [ -n "$pid" ] || continue
            if [ "$index" = "$COMFY_GPU_INDEX" ]; then
                case "$used_memory" in
                    ''|*[!0-9]*) return 1 ;;
                esac
                if [ "$used_memory" -le "$COMFY_MAX_IDLE_USED_MIB" ]; then
                    args=$(ps -p "$pid" -o args= 2>/dev/null || true)
                    queue=$(curl -fsS --max-time 2 "http://127.0.0.1:$COMFY_PORT/queue" 2>/dev/null | tr -d '[:space:]' || true)
                    case "$args" in
                        *main.py*"--port $COMFY_PORT"*|*main.py*"--port=$COMFY_PORT"*)
                            case "$queue" in
                                *'"queue_running":[]'*'"queue_pending":[]'*) continue ;;
                            esac
                            ;;
                    esac
                fi
            fi
            return 1
        done <<< "$apps"
        count=$((count + 1))
    done <<< "$rows"
    [ "$count" -eq "$GPU_COUNT" ]
}

gpu_idle_stable() {
    local sample=0
    while [ "$sample" -lt "$IDLE_STABILITY_SAMPLES" ]; do
        [ -f "$STOP_FILE" ] && return 1
        [ -f "$PAUSE_FILE" ] && return 1
        if ! gpu_idle; then
            return 1
        fi
        sample=$((sample + 1))
        if [ "$sample" -lt "$IDLE_STABILITY_SAMPLES" ]; then
            sleep "$IDLE_STABILITY_POLL_SECONDS"
        fi
    done
    return 0
}

log() {
    mkdir -p -- "$(dirname -- "$LOG_FILE")"
    printf '%s %s\n' "$(date -Is)" "$*" >> "$LOG_FILE"
}

run_loop() {
    mkdir -p -- "$WATCH_ROOT"
    trap 'rm -f -- "$PID_FILE"' EXIT INT TERM
    log "idle watcher started pid=$$ poll=${POLL_SECONDS}s min_free_mib=$MIN_FREE_MIB"
    local last_state="unknown"
    while [ ! -f "$STOP_FILE" ]; do
        if terminal_result_is_newer_than_start; then
            log "campaign produced a terminal result; stopping idle watcher"
            break
        fi
        if service_is_live; then
            if [ "$last_state" != "campaign_running" ]; then
                log "project campaign is running"
                last_state="campaign_running"
            fi
            sleep "$POLL_SECONDS"
            continue
        fi
        if [ -f "$PAUSE_FILE" ]; then
            if [ "$last_state" != "operator_paused" ]; then
                log "operator pause is active; waiting without starting project service"
                last_state="operator_paused"
            fi
            sleep "$POLL_SECONDS"
            continue
        fi
        if gpu_idle_stable; then
            if [ "$last_state" != "idle" ]; then
                log "all GPUs passed idle gate; starting project service"
                last_state="idle"
            fi
            date +%s > "$START_MARKER"
            if bash "$SERVICE" start >> "$LOG_FILE" 2>&1; then
                log "project service start requested"
            else
                log "project service refused start; will retry"
            fi
        elif [ "$last_state" != "busy" ]; then
            log "GPU idle gate not satisfied; waiting without touching external jobs"
            last_state="busy"
        fi
        sleep "$POLL_SECONDS"
    done
    rm -f -- "$PID_FILE"
    log "idle watcher stopped"
}

case "$ACTION" in
    run)
        run_loop
        ;;
    status)
        pid=$(read_pid || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && pid_is_watcher "$pid"; then
            echo "idle-autostart=$WATCH_ROOT pid=$pid state=running"
        else
            echo "idle-autostart=$WATCH_ROOT pid=none state=missing"
        fi
        ;;
    start)
        mkdir -p -- "$WATCH_ROOT"
        pid=$(read_pid || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && pid_is_watcher "$pid"; then
            echo "already running pid=$pid"
            exit 0
        fi
        rm -f -- "$PID_FILE" "$STOP_FILE"
        nohup env \
            REMOTE_CAMPAIGN_REPO_ROOT="$REPO_ROOT" \
            REMOTE_CAMPAIGN_OUTPUT="$OUTPUT" \
            REMOTE_CAMPAIGN_PAUSE_FILE="$PAUSE_FILE" \
            bash "$0" run >> "$LOG_FILE" 2>&1 < /dev/null &
        pid=$!
        printf '%s\n' "$pid" > "$PID_FILE"
        echo "started idle-autostart pid=$pid log=$LOG_FILE"
        ;;
    stop)
        touch -- "$STOP_FILE"
        pid=$(read_pid || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && pid_is_watcher "$pid"; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
        echo "idle-autostart stop requested"
        ;;
esac
