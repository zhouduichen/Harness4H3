#!/usr/bin/env bash

# Keep the local ComfyUI API available only for an active benchmark lease.
# When COMFY_LEASE_FILE is set, the campaign's lease manager owns the exact
# marker and this launcher exits as soon as the lease is released.  That keeps
# an on-demand ComfyUI from becoming a permanent GPU occupant after a crash or
# a release race.  Without a lease file, preserve the explicit always-on
# launcher behavior for operators who invoke this script manually.

set -u

COMFY_ROOT=${COMFY_ROOT:-/home/intern/huangjiahao/ComfyUI}
COMFY_PYTHON=${COMFY_PYTHON:-/home/intern/miniconda3/envs/comfy/bin/python}
COMFY_GPU_INDEX=${COMFY_GPU_INDEX:-0}
COMFY_HOST=${COMFY_HOST:-127.0.0.1}
COMFY_PORT=${COMFY_PORT:-8188}
COMFY_LOG=${COMFY_LOG:-/data/models/MiniMax-H3/harness4h3/comfyui.log}
COMFY_LEASE_FILE=${COMFY_LEASE_FILE:-}
# A stale campaign marker must not keep the evaluator alive forever after an
# SSH/session crash.  The campaign rewrites the marker at every benchmark
# boundary; the generous default covers a long H3 task while bounding orphan
# GPU ownership.
COMFY_LEASE_MAX_AGE_SECONDS=${COMFY_LEASE_MAX_AGE_SECONDS:-21600}
RESTART_SECONDS=${COMFY_RESTART_SECONDS:-5}
STOP_GRACE_SECONDS=${COMFY_STOP_GRACE_SECONDS:-20}
child_pid=""

lease_active() {
    [ -z "$COMFY_LEASE_FILE" ] && return 0
    [ -f "$COMFY_LEASE_FILE" ] || return 1
    # A campaign crash can leave a young marker behind.  Do not keep the
    # evaluator GPU alive merely because the timestamp is within the TTL:
    # only the live campaign owner may renew this lease.  Legacy markers
    # without owner_pid retain the age-only behavior for compatibility.
    local owner_pid owner_state owner_command
    owner_pid=$(sed -n 's/.*"owner_pid"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$COMFY_LEASE_FILE" 2>/dev/null | head -n 1)
    if [ -n "$owner_pid" ]; then
        owner_state=$(ps -p "$owner_pid" -o stat= 2>/dev/null | tr -d '[:space:]')
        owner_command=$(ps -p "$owner_pid" -o args= 2>/dev/null || true)
        case "$owner_state" in
            ''|Z*) return 1 ;;
        esac
        case "$owner_command" in
            *run_overnight_controller.py*|*remote-campaign-supervisor.sh*) ;;
            *) return 1 ;;
        esac
    fi
    local mtime now age
    mtime=$(stat -c %Y -- "$COMFY_LEASE_FILE" 2>/dev/null || true)
    case "$mtime" in
        ''|*[!0-9]*) return 0 ;;
    esac
    now=$(date +%s)
    age=$((now - mtime))
    [ "$age" -lt 0 ] || [ "$age" -le "$COMFY_LEASE_MAX_AGE_SECONDS" ]
}

stop_child() {
    if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM "$child_pid" 2>/dev/null || true
        local deadline=$((SECONDS + STOP_GRACE_SECONDS))
        while kill -0 "$child_pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
            sleep 1
        done
        if kill -0 "$child_pid" 2>/dev/null; then
            # Scope the forced cleanup to the exact child launched above.
            kill -KILL "$child_pid" 2>/dev/null || true
        fi
        wait "$child_pid" 2>/dev/null || true
    fi
    exit 143
}

trap stop_child TERM INT

mkdir -p "$(dirname "$COMFY_LOG")"

while true; do
    if ! lease_active; then
        stop_child
    fi

    if pgrep -f "main[.]py.*--port[ =]$COMFY_PORT" >/dev/null 2>&1; then
        sleep 5
        continue
    fi

    if curl -fsS --max-time 2 "http://$COMFY_HOST:$COMFY_PORT/system_stats" >/dev/null 2>&1; then
        sleep 5
        continue
    fi

    if [ ! -x "$COMFY_PYTHON" ]; then
        printf '%s ComfyUI launcher: Python not executable: %s\n' "$(date -Is)" "$COMFY_PYTHON" >> "$COMFY_LOG"
        sleep "$RESTART_SECONDS"
        continue
    fi

    printf '%s ComfyUI launcher: starting on GPU%s\n' "$(date -Is)" "$COMFY_GPU_INDEX" >> "$COMFY_LOG"
    (
        cd "$COMFY_ROOT" || exit 1
        exec env CUDA_VISIBLE_DEVICES="$COMFY_GPU_INDEX" "$COMFY_PYTHON" main.py \
            --listen "$COMFY_HOST" --port "$COMFY_PORT"
    ) >> "$COMFY_LOG" 2>&1 &
    child_pid=$!
    # Do not block forever in wait(2): a stale lease must be able to reclaim
    # the GPU while ComfyUI is still serving or stuck in shutdown.
    while kill -0 "$child_pid" 2>/dev/null; do
        if ! lease_active; then
            stop_child
        fi
        sleep 5
    done
    wait "$child_pid" 2>/dev/null || true
    child_pid=""
    if ! lease_active; then
        exit 0
    fi
    printf '%s ComfyUI launcher: child exited; retrying\n' "$(date -Is)" >> "$COMFY_LOG"
    sleep "$RESTART_SECONDS"
done
