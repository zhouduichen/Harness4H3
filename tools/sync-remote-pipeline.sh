#!/usr/bin/env bash
set -euo pipefail

# Copy only the trusted runtime surface into a remote staging directory.
# Activation and service start remain separate operations: a reconnect or a
# repeated sync never touches a live campaign or starts a GPU process.
SOURCE_ROOT="${REMOTE_PIPELINE_SOURCE_ROOT:-$(cd "$(dirname -- "$0")/.." && pwd)}"
REMOTE_HOST="${REMOTE_PIPELINE_HOST:-Jiayu-intern}"
SSH_PORT="${REMOTE_PIPELINE_SSH_PORT:-30902}"
REMOTE_REPO_ROOT="${REMOTE_CAMPAIGN_REPO_ROOT:-/home/intern/huangjiahao/Harness4H3-rsi}"
STAGE_ROOT="${REMOTE_PIPELINE_STAGE_ROOT:-$REMOTE_REPO_ROOT/work/remote-pipeline-v2-20260918}"
CONNECT_TIMEOUT="${REMOTE_PIPELINE_CONNECT_TIMEOUT:-8}"

case "$SSH_PORT" in
    ''|*[!0-9]*) echo "invalid REMOTE_PIPELINE_SSH_PORT: $SSH_PORT" >&2; exit 2 ;;
esac
if [ "$SSH_PORT" -lt 1 ] || [ "$SSH_PORT" -gt 65535 ]; then
    echo "REMOTE_PIPELINE_SSH_PORT must be between 1 and 65535" >&2
    exit 2
fi

files=(
    "harness4h3/remote/__init__.py"
    "harness4h3/remote/config.py"
    "harness4h3/remote/comfyui_lease.py"
    "harness4h3/remote/lane_packer.py"
    "harness4h3/remote/pipeline.py"
    "harness4h3/remote/power.py"
    "harness4h3/remote/round_gate.py"
    "harness4h3/remote/importer.py"
    "harness4h3/remote/scheduler.py"
    "harness4h3/remote/ssh.py"
    "harness4h3/benchmark/capabilities.py"
    "harness4h3/controller/context.py"
    "harness4h3/controller/provider.py"
    "harness4h3/controller/schemas.py"
    "harness4h3/controller/round_policy.py"
    "harness4h3/memory/discovery_digest.py"
    "harness4h3/memory/observation.py"
    "research/experiments/remote_h3_closed_loop.py"
    "configs/controller.yaml"
    "tools/activate-remote-pipeline.sh"
    "tools/run_overnight_controller.py"
    "tools/h3_checkpoint_stage.py"
    "tools/h3_real_train_worker.py"
    "tools/controller-wait-launch.sh"
    "tools/comfyui-wait-launch.sh"
    "tools/remote-campaign-service.sh"
    "tools/remote-campaign-supervisor.sh"
    "tools/remote-validation-window.sh"
    "tools/summarize_remote_validation.py"
    "tools/remote-idle-autostart.sh"
    "tools/remote-pipeline-handoff.sh"
    "configs/a1-worker.l40x4-rsi-staging.json"
    "configs/remote-l40-h3-rsi-overnight.yaml"
)

manifest="$(mktemp -t harness4h3-remote-pipeline.XXXXXX)"
cleanup() {
    rm -f -- "$manifest"
}
trap cleanup EXIT

for relative in "${files[@]}"; do
    source="$SOURCE_ROOT/$relative"
    if [ ! -f "$source" ]; then
        echo "source file is missing: $source" >&2
        exit 1
    fi
    printf '%s\n' "$relative" >> "$manifest"
done

ssh_args=(-p "$SSH_PORT" -o BatchMode=yes -o ConnectTimeout="$CONNECT_TIMEOUT")
echo "preparing remote staging directory: $REMOTE_HOST:$STAGE_ROOT"
ssh "${ssh_args[@]}" "$REMOTE_HOST" "mkdir -p -- '$STAGE_ROOT'"

rsync -a --files-from="$manifest" \
    -e "ssh -p $SSH_PORT -o BatchMode=yes -o ConnectTimeout=$CONNECT_TIMEOUT" \
    "$SOURCE_ROOT/" "$REMOTE_HOST:$STAGE_ROOT/"

echo "staged ${#files[@]} runtime files at $REMOTE_HOST:$STAGE_ROOT"
echo "next remote-only step: REMOTE_PIPELINE_STAGE_ROOT='$STAGE_ROOT' bash '$STAGE_ROOT/tools/remote-pipeline-handoff.sh'"
