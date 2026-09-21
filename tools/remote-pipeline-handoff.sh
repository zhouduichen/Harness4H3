#!/usr/bin/env bash
set -euo pipefail

# Wait on the existing campaign's lock, then perform the staged v2 handoff
# without a Codex process remaining connected.  This watcher is CPU-only and
# never manipulates a GPU process while the old campaign owns the lock.
REPO_ROOT="${REMOTE_CAMPAIGN_REPO_ROOT:-/home/intern/huangjiahao/Harness4H3-rsi}"
STAGE_ROOT="${REMOTE_PIPELINE_STAGE_ROOT:-$REPO_ROOT/work/remote-pipeline-v2-20260918}"
OUTPUT="${REMOTE_CAMPAIGN_OUTPUT:-$REPO_ROOT/var/remote-h3-controller-20260914}"
LOCK_FILE="$OUTPUT/.overnight-controller.lock"
PAUSE_FILE="${REMOTE_CAMPAIGN_PAUSE_FILE:-$OUTPUT/.operator-paused}"
LOG_FILE="${REMOTE_PIPELINE_HANDOFF_LOG:-$REPO_ROOT/work/remote-pipeline-v2-20260918/handoff.log}"
PID_FILE="${REMOTE_PIPELINE_HANDOFF_PID:-$REPO_ROOT/work/remote-pipeline-v2-20260918/handoff.pid}"
GPU_COUNT="${REMOTE_PIPELINE_HANDOFF_GPU_COUNT:-4}"
MIN_FREE_MIB="${REMOTE_PIPELINE_HANDOFF_MIN_FREE_MIB:-20000}"
GPU_POLL_S="${REMOTE_PIPELINE_HANDOFF_GPU_POLL_S:-30}"
IDLE_STABILITY_SAMPLES="${REMOTE_PIPELINE_HANDOFF_IDLE_STABILITY_SAMPLES:-3}"
IDLE_STABILITY_POLL_S="${REMOTE_PIPELINE_HANDOFF_IDLE_STABILITY_POLL_S:-5}"
COMFY_GPU_INDEX="${REMOTE_PIPELINE_HANDOFF_COMFY_GPU_INDEX:-0}"
COMFY_PORT="${REMOTE_PIPELINE_HANDOFF_COMFY_PORT:-8188}"
COMFY_MAX_IDLE_USED_MIB="${REMOTE_PIPELINE_HANDOFF_COMFY_MAX_IDLE_USED_MIB:-2048}"

mkdir -p -- "$(dirname "$LOG_FILE")" "$(dirname "$PID_FILE")"
printf '%s\n' "$$" > "$PID_FILE"
cleanup() {
    rm -f -- "$PID_FILE"
}

trap cleanup EXIT
trap 'exit 0' INT TERM

operator_pause_is_active() {
    [ -f "$PAUSE_FILE" ]
}

if operator_pause_is_active; then
    printf '%s operator pause is active; leaving staged pipeline untouched\n' "$(date -Is)" >> "$LOG_FILE"
    exit 0
fi

log_waiting() {
    printf '%s handoff watcher pid=%s waiting: lock=%s exact_supervisor=%s\n' \
        "$(date -Is)" "$$" "$1" "$2" >> "$LOG_FILE"
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

lock_is_held() {
    command -v flock >/dev/null 2>&1 || return 1
    ! flock -n "$LOCK_FILE" -c true 2>/dev/null
}

overnight_controller_processes() {
    # The restart wrapper can retain the runner path in its bash command line
    # while waiting for the Python child. Only the real Python process owns
    # the campaign execution boundary.
    ps -eo pid=,comm=,args= 2>/dev/null | awk '
        $2 ~ /^python([0-9.]*)?$/ &&
        $0 ~ /(^|[[:space:]])tools\/run_overnight_controller[.]py([[:space:]]|$)/ {print}
    '
}

last_log_at=0
while lock_is_held; do
    if operator_pause_is_active; then
        printf '%s operator pause appeared while waiting for campaign lock; leaving staged pipeline untouched\n' "$(date -Is)" >> "$LOG_FILE"
        exit 0
    fi
    now=$(date +%s)
    if [ "$((now - last_log_at))" -ge 300 ]; then
        log_waiting held present
        last_log_at="$now"
    fi
    sleep "${REMOTE_PIPELINE_HANDOFF_POLL_S:-60}"
done

# A process can briefly outlive the lock-file release while its parent is
# unwinding.  Wait for the exact supervisor command before changing code.
log_waiting released present
while overnight_controller_processes | grep -q .; do
    if operator_pause_is_active; then
        printf '%s operator pause appeared while waiting for Controller process; leaving staged pipeline untouched\n' "$(date -Is)" >> "$LOG_FILE"
        exit 0
    fi
    now=$(date +%s)
    if [ "$((now - last_log_at))" -ge 300 ]; then
        log_waiting released present
        last_log_at="$now"
    fi
    sleep "${REMOTE_PIPELINE_HANDOFF_POLL_S:-60}"
done

# Do not activate the new launcher merely because the old lock disappeared.
# A shared host can release the campaign lock while another user's GPU job is
# still running.  Wait on the same fail-closed four-card gate as the idle
# autostart watcher; this handoff process remains CPU-only until then.
last_log_at=0
while ! gpu_idle; do
    if operator_pause_is_active; then
        printf '%s operator pause appeared before idle gate; leaving staged pipeline untouched\n' "$(date -Is)" >> "$LOG_FILE"
        exit 0
    fi
    now=$(date +%s)
    if [ "$((now - last_log_at))" -ge 300 ]; then
        printf '%s handoff waiting for all %s GPUs to be idle (min_free_mib=%s)\n' \
            "$(date -Is)" "$GPU_COUNT" "$MIN_FREE_MIB" >> "$LOG_FILE"
        last_log_at="$now"
    fi
    sleep "$GPU_POLL_S"
done

# Close the last race between the final idle query and activation.  A foreign
# job can appear immediately after a single green sample; require a short
# stable window and restart the sample sequence if any card becomes busy.
stable_samples=0
while [ "$stable_samples" -lt "$IDLE_STABILITY_SAMPLES" ]; do
    if operator_pause_is_active; then
        printf '%s operator pause appeared during idle stability gate; leaving staged pipeline untouched\n' "$(date -Is)" >> "$LOG_FILE"
        exit 0
    fi
    if gpu_idle; then
        stable_samples=$((stable_samples + 1))
        printf '%s handoff idle stability sample %s/%s\n' \
            "$(date -Is)" "$stable_samples" "$IDLE_STABILITY_SAMPLES" >> "$LOG_FILE"
        if [ "$stable_samples" -lt "$IDLE_STABILITY_SAMPLES" ]; then
            sleep "$IDLE_STABILITY_POLL_S"
        fi
    else
        stable_samples=0
        printf '%s handoff idle stability reset by a newly busy GPU\n' "$(date -Is)" >> "$LOG_FILE"
        sleep "$GPU_POLL_S"
    fi
done
if operator_pause_is_active; then
    printf '%s operator pause appeared after idle stability gate; leaving staged pipeline untouched\n' "$(date -Is)" >> "$LOG_FILE"
    exit 0
fi
printf '%s all GPUs passed handoff idle gate (stable); activating staged pipeline\n' "$(date -Is)" >> "$LOG_FILE"

{
    printf '%s old campaign ended; activating staged pipeline\n' "$(date -Is)"
    REMOTE_CAMPAIGN_REPO_ROOT="$REPO_ROOT" \
        REMOTE_PIPELINE_STAGE_ROOT="$STAGE_ROOT" \
        REMOTE_CAMPAIGN_OUTPUT="$OUTPUT" \
        bash "$STAGE_ROOT/tools/activate-remote-pipeline.sh"
    REMOTE_CAMPAIGN_REPO_ROOT="$REPO_ROOT" \
        REMOTE_CAMPAIGN_OUTPUT="$OUTPUT" \
        bash "$REPO_ROOT/tools/remote-campaign-service.sh" start
} >> "$LOG_FILE" 2>&1
