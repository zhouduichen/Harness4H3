#!/usr/bin/env bash
set -euo pipefail

# Run a bounded, evidence-producing remote campaign window. The default path
# is deliberately read-only: an operator must pass --start after explicitly
# removing the campaign pause marker.

REPO_ROOT="${REMOTE_CAMPAIGN_REPO_ROOT:-/home/intern/huangjiahao/Harness4H3-rsi}"
PYTHON="${REMOTE_CAMPAIGN_PYTHON:-/home/intern/miniconda3/envs/comfy/bin/python}"
CONFIG="${REMOTE_CAMPAIGN_CONFIG:-$REPO_ROOT/configs/remote-l40-h3-rsi-overnight.yaml}"
CAMPAIGN_OUTPUT="${REMOTE_CAMPAIGN_OUTPUT:-$REPO_ROOT/var/remote-h3-controller-20260914}"
SERVICE="$REPO_ROOT/tools/remote-campaign-service.sh"
SUMMARIZER="$REPO_ROOT/tools/summarize_remote_validation.py"
START=0
MAX_ITERATIONS="${REMOTE_VALIDATION_MAX_ITERATIONS:-2}"
MAX_RUNTIME_S="${REMOTE_VALIDATION_MAX_RUNTIME_S:-7200}"
POLL_INTERVAL_S="${REMOTE_VALIDATION_POLL_INTERVAL_S:-5}"
SAMPLE_INTERVAL_S="${REMOTE_VALIDATION_SAMPLE_INTERVAL_S:-2}"
EVIDENCE_ROOT="${REMOTE_VALIDATION_OUTPUT:-}"
EXPECTED_GPUS="${REMOTE_VALIDATION_EXPECTED_GPUS:-4}"
CLEANUP_WAIT_S="${REMOTE_VALIDATION_CLEANUP_WAIT_S:-120}"

usage() {
    cat <<'EOF'
usage: remote-validation-window.sh [--start] [options]

Without --start this command only reports the configured campaign status.
Options:
  --start                 explicitly start one bounded campaign window
  --max-iterations N      campaign iteration limit (default: 2)
  --max-runtime-s N       wall-clock limit before graceful pause (default: 7200)
  --poll-interval-s N     service status polling interval (default: 5)
  --sample-interval-s N   nvidia-smi sampling interval (default: 2)
  --output-root PATH      evidence directory
  --help                  show this message
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --start) START=1 ;;
        --max-iterations)
            shift
            [ "$#" -gt 0 ] || { echo "--max-iterations requires a value" >&2; exit 2; }
            MAX_ITERATIONS="$1"
            ;;
        --max-runtime-s)
            shift
            [ "$#" -gt 0 ] || { echo "--max-runtime-s requires a value" >&2; exit 2; }
            MAX_RUNTIME_S="$1"
            ;;
        --poll-interval-s)
            shift
            [ "$#" -gt 0 ] || { echo "--poll-interval-s requires a value" >&2; exit 2; }
            POLL_INTERVAL_S="$1"
            ;;
        --sample-interval-s)
            shift
            [ "$#" -gt 0 ] || { echo "--sample-interval-s requires a value" >&2; exit 2; }
            SAMPLE_INTERVAL_S="$1"
            ;;
        --output-root)
            shift
            [ "$#" -gt 0 ] || { echo "--output-root requires a path" >&2; exit 2; }
            EVIDENCE_ROOT="$1"
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

positive_integer() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ "$1" -gt 0 ]
}

positive_number() {
    awk -v value="$1" 'BEGIN { exit !(value + 0 > 0) }'
}

if ! positive_integer "$MAX_ITERATIONS" || ! positive_integer "$MAX_RUNTIME_S" || ! positive_integer "$EXPECTED_GPUS" || ! positive_integer "$CLEANUP_WAIT_S"; then
    echo "iteration, runtime, GPU count, and cleanup wait values must be positive integers" >&2
    exit 2
fi
if ! positive_number "$POLL_INTERVAL_S" || ! positive_number "$SAMPLE_INTERVAL_S"; then
    echo "poll and sample intervals must be positive numbers" >&2
    exit 2
fi

if [ "$START" -eq 0 ]; then
    env \
        "REMOTE_CAMPAIGN_REPO_ROOT=$REPO_ROOT" \
        "REMOTE_CAMPAIGN_PYTHON=$PYTHON" \
        "REMOTE_CAMPAIGN_CONFIG=$CONFIG" \
        "REMOTE_CAMPAIGN_OUTPUT=$CAMPAIGN_OUTPUT" \
        bash "$SERVICE" status 2>&1 || true
    exit 0
fi

if [ -z "$EVIDENCE_ROOT" ]; then
    EVIDENCE_ROOT="$REPO_ROOT/work/remote-validation-window-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p -- "$(dirname -- "$EVIDENCE_ROOT")"
EVIDENCE_ROOT="$(cd "$(dirname -- "$EVIDENCE_ROOT")" && pwd)/$(basename -- "$EVIDENCE_ROOT")"
PAUSE_FILE="${REMOTE_CAMPAIGN_PAUSE_FILE:-$CAMPAIGN_OUTPUT/.operator-paused}"
EVENT_LOG="$CAMPAIGN_OUTPUT/controller-events.jsonl"
STATE_FILE="$CAMPAIGN_OUTPUT/campaign_state.json"
RESULT_FILE="$CAMPAIGN_OUTPUT/overnight-result.json"
TELEMETRY_FILE="$EVIDENCE_ROOT/gpu-telemetry.csv"
STATUS_FILE="$EVIDENCE_ROOT/status-timeline.tsv"
PRE_GPU_FILE="$EVIDENCE_ROOT/gpu-before.csv"
POST_GPU_FILE="$EVIDENCE_ROOT/gpu-after.csv"
PROCESS_BEFORE_FILE="$EVIDENCE_ROOT/compute-processes-before.txt"
PROCESS_AFTER_FILE="$EVIDENCE_ROOT/compute-processes-after.txt"
MANIFEST_FILE="$EVIDENCE_ROOT/window-manifest.json"

SERVICE_ENV=(
    "REMOTE_CAMPAIGN_REPO_ROOT=$REPO_ROOT"
    "REMOTE_CAMPAIGN_PYTHON=$PYTHON"
    "REMOTE_CAMPAIGN_CONFIG=$CONFIG"
    "REMOTE_CAMPAIGN_OUTPUT=$CAMPAIGN_OUTPUT"
    "REMOTE_CAMPAIGN_PAUSE_FILE=$PAUSE_FILE"
    "REMOTE_CAMPAIGN_MAX_ITERATIONS=$MAX_ITERATIONS"
)

service_cmd() {
    env "${SERVICE_ENV[@]}" bash "$SERVICE" "$@"
}

service_status() {
    service_cmd status 2>&1 || true
}

status_state() {
    printf '%s\n' "$1" | awk '
        match($0, /state=[^[:space:]]+/) {
            value = substr($0, RSTART + 6, RLENGTH - 6)
            print value
            exit
        }
    '
}

state_is_live() {
    case "$1" in
        running|stop_requested|running_without_service_pid) return 0 ;;
        *) return 1 ;;
    esac
}

file_size() {
    if [ -f "$1" ]; then
        stat -c %s -- "$1" 2>/dev/null || stat -f %z -- "$1" 2>/dev/null || printf '0\n'
    else
        printf '0\n'
    fi
}

gpu_snapshot() {
    nvidia-smi --query-gpu=index,power.draw,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
}

compute_process_snapshot() {
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
}

nonempty_line_count() {
    awk 'NF && $0 !~ /^No running processes found/ { count += 1 } END { print count + 0 }'
}

sampler_loop() {
    printf 'timestamp,index,power_w,utilization_gpu_pct,memory_used_mib,memory_total_mib\n'
    while :; do
        timestamp="$(date +%s.%N)"
        gpu_snapshot | awk -F, -v timestamp="$timestamp" '{
            for (i = 1; i <= NF; i++) {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
            }
            if (NF == 5 && $1 ~ /^[0-9]+$/) {
                printf "%s,%s,%s,%s,%s,%s\n", timestamp, $1, $2, $3, $4, $5
            }
        }'
        sleep "$SAMPLE_INTERVAL_S"
    done
}

write_manifest() {
    local ended_at="$1" terminal_reason="$2" status_after="$3" state_after="$4" event_start_bytes="$5"
    "$PYTHON" - "$MANIFEST_FILE" "$EVIDENCE_ROOT" "$CAMPAIGN_OUTPUT" "$EVENT_LOG" "$STATE_FILE" "$RESULT_FILE" \
        "$TELEMETRY_FILE" "$STATUS_FILE" "$PRE_GPU_FILE" "$POST_GPU_FILE" "$PROCESS_BEFORE_FILE" "$PROCESS_AFTER_FILE" \
        "$STARTED_AT" "$ended_at" "$MAX_ITERATIONS" "$MAX_RUNTIME_S" "$EXPECTED_GPUS" "$terminal_reason" \
        "$status_after" "$state_after" "$event_start_bytes" <<'PY'
import json
import sys
from pathlib import Path

(
    manifest_path, evidence_root, campaign_output, event_log, state_file,
    result_file, telemetry_file, status_file, pre_gpu_file, post_gpu_file,
    process_before_file, process_after_file, started_at, ended_at,
    max_iterations, max_runtime_s, expected_gpus, terminal_reason,
    status_after, state_after, event_start_bytes,
) = sys.argv[1:]
payload = {
    "schema_version": 1,
    "evidence_root": evidence_root,
    "campaign_output": campaign_output,
    "event_log": event_log,
    "state_file": state_file,
    "result_file": result_file,
    "telemetry_file": telemetry_file,
    "status_timeline": status_file,
    "gpu_before": pre_gpu_file,
    "gpu_after": post_gpu_file,
    "compute_processes_before": process_before_file,
    "compute_processes_after": process_after_file,
    "started_at_epoch": float(started_at),
    "ended_at_epoch": float(ended_at),
    "max_iterations": int(max_iterations),
    "max_runtime_s": int(max_runtime_s),
    "expected_gpu_count": int(expected_gpus),
    "terminal_reason": terminal_reason,
    "campaign_started": True,
    "service_status_after": status_after,
    "service_state_after": state_after,
    "event_start_byte": int(event_start_bytes),
}
target = Path(manifest_path)
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_name(target.name + ".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
PY
}

mkdir -p -- "$EVIDENCE_ROOT"
[ -x "$SERVICE" ] || { echo "campaign service is missing or not executable: $SERVICE" >&2; exit 4; }
[ -f "$PAUSE_FILE" ] && { echo "campaign is operator-paused; resume it explicitly before --start: $PAUSE_FILE" >&2; exit 3; }
[ -f "$SUMMARIZER" ] || { echo "validation summarizer is missing: $SUMMARIZER" >&2; exit 4; }

initial_status="$(service_status)"
initial_state="$(status_state "$initial_status")"
if state_is_live "$initial_state"; then
    echo "campaign is already live; refusing a second validation window" >&2
    exit 3
fi

command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required for a real validation window" >&2; exit 4; }
compute_process_snapshot > "$PROCESS_BEFORE_FILE"
if [ "$(nonempty_line_count < "$PROCESS_BEFORE_FILE")" -ne 0 ]; then
    echo "GPU compute processes are present; refusing to touch a shared host" >&2
    exit 3
fi
gpu_snapshot > "$PRE_GPU_FILE"
if [ "$(nonempty_line_count < "$PRE_GPU_FILE")" -lt "$EXPECTED_GPUS" ]; then
    echo "fewer than $EXPECTED_GPUS GPUs were observed; refusing validation" >&2
    exit 3
fi

event_start_bytes="$(file_size "$EVENT_LOG")"
STARTED_AT="$(date +%s)"
STARTED=1
SAMPLER_PID=""
TERMINAL_REASON="unknown"
CLEANED=0

cleanup() {
    local exit_status=$?
    [ "$CLEANED" -eq 0 ] || exit "$exit_status"
    CLEANED=1
    trap - EXIT INT TERM
    service_cmd pause > "$EVIDENCE_ROOT/final-pause-request.txt" 2>&1 || true
    local wait_count=0 current_status current_state
    while [ "$wait_count" -lt "$CLEANUP_WAIT_S" ]; do
        current_status="$(service_status)"
        current_state="$(status_state "$current_status")"
        if ! state_is_live "$current_state"; then
            break
        fi
        sleep 1
        wait_count=$((wait_count + 1))
    done
    if [ -n "$SAMPLER_PID" ] && kill -0 "$SAMPLER_PID" 2>/dev/null; then
        kill -TERM "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
    fi
    gpu_snapshot > "$POST_GPU_FILE" 2> "$EVIDENCE_ROOT/gpu-after.error" || true
    compute_process_snapshot > "$PROCESS_AFTER_FILE" 2> "$EVIDENCE_ROOT/compute-processes-after.error" || true
    current_status="$(service_status)"
    current_state="$(status_state "$current_status")"
    printf '%s\t%s\n' "$(date +%s)" "${current_state:-unknown}" >> "$STATUS_FILE"
    ended_at="$(date +%s)"
    write_manifest "$ended_at" "$TERMINAL_REASON" "$current_status" "$current_state" "$event_start_bytes"
    "$PYTHON" "$SUMMARIZER" --manifest "$MANIFEST_FILE" || true
    exit "$exit_status"
}
trap cleanup EXIT INT TERM

sampler_loop > "$TELEMETRY_FILE" 2> "$EVIDENCE_ROOT/gpu-telemetry.error" &
SAMPLER_PID=$!
start_output=""
if ! start_output="$(service_cmd start 2>&1)"; then
    printf '%s\n' "$start_output" > "$EVIDENCE_ROOT/service-start.txt"
    TERMINAL_REASON="service_start_refused"
    exit 4
fi
printf '%s\n' "$initial_status" > "$EVIDENCE_ROOT/service-before.txt"
printf '%s\n' "$start_output" > "$EVIDENCE_ROOT/service-start.txt"

deadline=$((STARTED_AT + MAX_RUNTIME_S))
while :; do
    current_status="$(service_status)"
    current_state="$(status_state "$current_status")"
    printf '%s\t%s\n' "$(date +%s)" "${current_state:-unknown}" >> "$STATUS_FILE"
    if ! state_is_live "$current_state"; then
        if [ -f "$RESULT_FILE" ]; then
            TERMINAL_REASON="campaign_terminal_result"
        else
            TERMINAL_REASON="campaign_exited_without_result"
        fi
        break
    fi
    now="$(date +%s)"
    if [ "$now" -ge "$deadline" ]; then
        TERMINAL_REASON="wall_clock_limit_graceful_stop"
        service_cmd pause > "$EVIDENCE_ROOT/timeout-pause-request.txt" 2>&1 || true
        break
    fi
    sleep "$POLL_INTERVAL_S"
done

exit 0
