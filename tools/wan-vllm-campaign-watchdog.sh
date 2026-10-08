#!/usr/bin/env bash
set -u

# Detached lifecycle wrapper for the real Wan vLLM campaign.  It owns only
# the configured campaign child; experiment state and recovery stay in
# wan_vllm_campaign.py and TrainerEngine checkpoints.

WAN_CAMPAIGN_PYTHON="${WAN_CAMPAIGN_PYTHON:-}"
WAN_CAMPAIGN_SCRIPT="${WAN_CAMPAIGN_SCRIPT:-}"
WAN_CAMPAIGN_CONFIG="${WAN_CAMPAIGN_CONFIG:-}"
WAN_CAMPAIGN_OUTPUT="${WAN_CAMPAIGN_OUTPUT:-}"
WAN_CAMPAIGN_MAX_ROUNDS="${WAN_CAMPAIGN_MAX_ROUNDS:-2}"
WAN_CAMPAIGN_RESTART_LIMIT="${WAN_CAMPAIGN_RESTART_LIMIT:-3}"
WAN_CAMPAIGN_BACKOFF_SECONDS="${WAN_CAMPAIGN_BACKOFF_SECONDS:-30}"
WAN_CAMPAIGN_STOP_FILE="${WAN_CAMPAIGN_STOP_FILE:-$WAN_CAMPAIGN_OUTPUT/.watchdog-stop}"
WAN_CAMPAIGN_LOCK_FILE="${WAN_CAMPAIGN_LOCK_FILE:-$WAN_CAMPAIGN_OUTPUT/.watchdog.lock}"
WAN_CAMPAIGN_CHILD_PID_FILE="${WAN_CAMPAIGN_CHILD_PID_FILE:-$WAN_CAMPAIGN_OUTPUT/.campaign-child.pid}"
WAN_CAMPAIGN_LOG="${WAN_CAMPAIGN_LOG:-$WAN_CAMPAIGN_OUTPUT/watchdog.log}"
WAN_CAMPAIGN_STOP_GRACE_SECONDS="${WAN_CAMPAIGN_STOP_GRACE_SECONDS:-30}"

die() {
    echo "wan watchdog: $*" >&2
    exit 2
}

case "$WAN_CAMPAIGN_MAX_ROUNDS" in ''|*[!0-9]*) die "WAN_CAMPAIGN_MAX_ROUNDS must be an integer" ;; esac
case "$WAN_CAMPAIGN_RESTART_LIMIT" in ''|*[!0-9]*) die "WAN_CAMPAIGN_RESTART_LIMIT must be an integer" ;; esac
case "$WAN_CAMPAIGN_BACKOFF_SECONDS" in ''|*[!0-9]*) die "WAN_CAMPAIGN_BACKOFF_SECONDS must be an integer" ;; esac
case "$WAN_CAMPAIGN_STOP_GRACE_SECONDS" in ''|*[!0-9]*) die "WAN_CAMPAIGN_STOP_GRACE_SECONDS must be an integer" ;; esac
[ -n "$WAN_CAMPAIGN_PYTHON" ] || die "WAN_CAMPAIGN_PYTHON is required"
[ -x "$WAN_CAMPAIGN_PYTHON" ] || die "trusted Python is not executable: $WAN_CAMPAIGN_PYTHON"
[ -n "$WAN_CAMPAIGN_SCRIPT" ] || die "WAN_CAMPAIGN_SCRIPT is required"
[ -f "$WAN_CAMPAIGN_SCRIPT" ] || die "trusted campaign script is missing: $WAN_CAMPAIGN_SCRIPT"
[ -n "$WAN_CAMPAIGN_CONFIG" ] || die "WAN_CAMPAIGN_CONFIG is required"
[ -f "$WAN_CAMPAIGN_CONFIG" ] || die "trusted campaign config is missing: $WAN_CAMPAIGN_CONFIG"
[ -n "$WAN_CAMPAIGN_OUTPUT" ] || die "WAN_CAMPAIGN_OUTPUT is required"

mkdir -p -- "$WAN_CAMPAIGN_OUTPUT" "$(dirname -- "$WAN_CAMPAIGN_LOG")" "$(dirname -- "$WAN_CAMPAIGN_CHILD_PID_FILE")"

log() {
    printf '%s %s\n' "$(date -Is)" "$*" >> "$WAN_CAMPAIGN_LOG"
}

lock_dir="${WAN_CAMPAIGN_LOCK_FILE}.d"
lock_mode=""
if command -v flock >/dev/null 2>&1; then
    exec 9>"$WAN_CAMPAIGN_LOCK_FILE"
    if ! flock -n 9; then
        log "another Wan watchdog owns lock=$WAN_CAMPAIGN_LOCK_FILE"
        exit 3
    fi
    lock_mode="flock"
else
    if ! mkdir -- "$lock_dir" 2>/dev/null; then
        log "another Wan watchdog owns lock=$lock_dir"
        exit 3
    fi
    lock_mode="mkdir"
fi

child_pid=""
stop_requested=0

clear_child_pid() {
    rm -f -- "$WAN_CAMPAIGN_CHILD_PID_FILE"
}

write_child_pid() {
    printf '%s\n' "$child_pid" > "$WAN_CAMPAIGN_CHILD_PID_FILE"
}

child_matches() {
    [ -n "$child_pid" ] || return 1
    local command
    command=$(ps -p "$child_pid" -o args= 2>/dev/null || true)
    [[ "$command" == *"$WAN_CAMPAIGN_SCRIPT"* && "$command" == *"$WAN_CAMPAIGN_CONFIG"* ]]
}

kill_tree() {
    local root="$1" signal="$2" child
    for child in $(pgrep -P "$root" 2>/dev/null || true); do
        kill_tree "$child" "$signal"
    done
    kill -"$signal" "$root" 2>/dev/null || true
}

terminate_child() {
    if [ -z "$child_pid" ] || ! kill -0 "$child_pid" 2>/dev/null; then
        clear_child_pid
        return 0
    fi
    if ! child_matches; then
        log "refusing to signal unexpected child pid=$child_pid"
        return 1
    fi
    log "stopping campaign child pid=$child_pid"
    kill_tree "$child_pid" TERM
    local deadline=$((SECONDS + WAN_CAMPAIGN_STOP_GRACE_SECONDS))
    while kill -0 "$child_pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
        sleep 1
    done
    if kill -0 "$child_pid" 2>/dev/null; then
        log "campaign child did not stop within ${WAN_CAMPAIGN_STOP_GRACE_SECONDS}s; force stopping its process tree"
        kill_tree "$child_pid" KILL
    fi
    clear_child_pid
}

on_signal() {
    stop_requested=1
    touch -- "$WAN_CAMPAIGN_STOP_FILE"
    terminate_child || true
    exit 0
}

trap on_signal TERM INT
trap 'clear_child_pid; [ "$lock_mode" = "mkdir" ] && rmdir -- "$lock_dir" 2>/dev/null || true' EXIT

if [ -f "$WAN_CAMPAIGN_STOP_FILE" ]; then
    log "operator stop marker already exists; watchdog will not launch campaign"
    exit 0
fi

attempt=0
while :; do
    if [ "$stop_requested" -ne 0 ] || [ -f "$WAN_CAMPAIGN_STOP_FILE" ]; then
        log "stop requested before launch"
        exit 0
    fi
    if [ -f "$WAN_CAMPAIGN_OUTPUT/campaign-result.json" ]; then
        log "final campaign result already exists; watchdog is complete"
        exit 0
    fi

    log "launching Wan campaign attempt=$((attempt + 1)) max_rounds=$WAN_CAMPAIGN_MAX_ROUNDS"
    "$WAN_CAMPAIGN_PYTHON" "$WAN_CAMPAIGN_SCRIPT" \
        --config "$WAN_CAMPAIGN_CONFIG" \
        --max-rounds "$WAN_CAMPAIGN_MAX_ROUNDS" &
    child_pid=$!
    write_child_pid
    wait "$child_pid"
    status=$?
    clear_child_pid
    child_pid=""

    if [ -f "$WAN_CAMPAIGN_OUTPUT/campaign-result.json" ]; then
        log "campaign exited status=$status with durable final result; watchdog stopping"
        exit "$status"
    fi
    if [ "$stop_requested" -ne 0 ] || [ -f "$WAN_CAMPAIGN_STOP_FILE" ]; then
        log "campaign exited status=$status after operator stop; watchdog stopping"
        exit 0
    fi

    attempt=$((attempt + 1))
    if [ "$attempt" -ge "$WAN_CAMPAIGN_RESTART_LIMIT" ]; then
        log "campaign exited status=$status without final result; restart limit exhausted"
        exit "$status"
    fi
    delay=$((WAN_CAMPAIGN_BACKOFF_SECONDS * attempt))
    log "campaign exited status=$status without final result; restarting after ${delay}s"
    sleep "$delay"
done
