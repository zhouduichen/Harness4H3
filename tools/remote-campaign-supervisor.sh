#!/usr/bin/env bash
set -euo pipefail

# Own exactly one campaign runner and release the project Controller as soon
# as that runner reaches any terminal result. This process is deliberately
# launched by remote-campaign-service.sh and never searches for foreign jobs.
REPO_ROOT="${REMOTE_CAMPAIGN_REPO_ROOT:-/home/intern/huangjiahao/Harness4H3-rsi}"
PYTHON="${REMOTE_CAMPAIGN_PYTHON:-/home/intern/miniconda3/envs/comfy/bin/python}"
CONFIG="${REMOTE_CAMPAIGN_CONFIG:-$REPO_ROOT/configs/remote-l40-h3-rsi-overnight.yaml}"
OUTPUT="${REMOTE_CAMPAIGN_OUTPUT:-$REPO_ROOT/var/remote-h3-controller-20260914}"
STOP_FILE="${REMOTE_CAMPAIGN_STOP_FILE:-$OUTPUT/.stop-requested}"
CONTROLLER_PORT="${REMOTE_CAMPAIGN_CONTROLLER_PORT:-8001}"
CONTROLLER_FALLBACK_PORTS="${REMOTE_CAMPAIGN_CONTROLLER_FALLBACK_PORTS:-8000}"
RESOURCE_POLL_INTERVAL_S="${REMOTE_CAMPAIGN_RESOURCE_POLL_INTERVAL_S:-15}"
MAX_ITERATIONS="${REMOTE_CAMPAIGN_MAX_ITERATIONS:-512}"
RUNNER="$REPO_ROOT/tools/run_overnight_controller.py"
SERVICE="$REPO_ROOT/tools/remote-campaign-service.sh"

case "$MAX_ITERATIONS" in
    ''|*[!0-9]*)
        echo "REMOTE_CAMPAIGN_MAX_ITERATIONS must be a positive integer" >&2
        exit 2
        ;;
esac
if [ "$MAX_ITERATIONS" -le 0 ]; then
    echo "REMOTE_CAMPAIGN_MAX_ITERATIONS must be a positive integer" >&2
    exit 2
fi

child_pid=""

kill_child_tree() {
    local root="$1" node
    for node in $(pgrep -P "$root" 2>/dev/null || true); do
        kill_child_tree "$node"
    done
    kill -TERM "$root" 2>/dev/null || true
}

cleanup() {
    if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
        kill_child_tree "$child_pid"
    fi
    bash "$SERVICE" cleanup-controller || true
}

trap cleanup EXIT INT TERM

bash "$SERVICE" ensure-controller
"$PYTHON" "$RUNNER" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --max-iterations "$MAX_ITERATIONS" \
    --on-server \
    --stop-file "$STOP_FILE" \
    --resource-poll-interval-s "$RESOURCE_POLL_INTERVAL_S" \
    --controller-remote-port "$CONTROLLER_PORT" \
    --controller-fallback-ports "$CONTROLLER_FALLBACK_PORTS" &
child_pid=$!
wait "$child_pid"
