#!/usr/bin/env bash
set -euo pipefail

# Activate a staged pipeline only after the campaign single-flight lock is
# released.  The script is intentionally conservative: it refuses to touch a
# live supervisor and keeps a timestamped backup of every replaced file.
REPO_ROOT="${REMOTE_CAMPAIGN_REPO_ROOT:-/home/intern/huangjiahao/Harness4H3-rsi}"
STAGE_ROOT="${REMOTE_PIPELINE_STAGE_ROOT:-$REPO_ROOT/work/remote-pipeline-v2-20260918}"
OUTPUT="${REMOTE_CAMPAIGN_OUTPUT:-$REPO_ROOT/var/remote-h3-controller-20260914}"
CAMPAIGN_ROOT="${REMOTE_CAMPAIGN_ROOT:-$REPO_ROOT/work/remote-h3-controller-20260914}"
LOCK_FILE="$OUTPUT/.overnight-controller.lock"
BACKUP_ROOT="$REPO_ROOT/work/remote-pipeline-backup-$(date +%Y%m%d-%H%M%S)-$$"
LAUNCHER_LOG="$REPO_ROOT/work/remote-h3-controller-20260914/controller-launcher.log"

lock_is_held() {
    command -v flock >/dev/null 2>&1 || return 1
    ! flock -n "$LOCK_FILE" -c true 2>/dev/null
}

kill_process_tree() {
    local root="$1" signal="$2" child
    for child in $(pgrep -P "$root" 2>/dev/null || true); do
        kill_process_tree "$child" "$signal"
    done
    kill -"$signal" "$root" 2>/dev/null || true
}

controller_launcher_matches() {
    local pid="$1" command
    command=$(ps -p "$pid" -o args= 2>/dev/null || true)
    [[ "$command" == *"$REPO_ROOT/tools/controller-wait-launch.sh"* ]]
}

stop_old_controller_launchers() {
    local pid
    for pid in $(pgrep -f "$REPO_ROOT/tools/controller-wait-launch.sh" 2>/dev/null || true); do
        if controller_launcher_matches "$pid"; then
            echo "stopping old campaign Controller launcher pid=$pid"
            kill_process_tree "$pid" TERM
        fi
    done
    local deadline=$((SECONDS + 30))
    while pgrep -f "$REPO_ROOT/tools/controller-wait-launch.sh" >/dev/null 2>&1 && [ "$SECONDS" -lt "$deadline" ]; do
        sleep 1
    done
    if pgrep -f "$REPO_ROOT/tools/controller-wait-launch.sh" >/dev/null 2>&1; then
        echo "old Controller launcher did not stop; refusing activation" >&2
        return 1
    fi
}

overnight_controller_processes() {
    # Match the real Python supervisor, not a shell wrapper whose command
    # string merely contains the runner path while it waits for a child.
    ps -eo pid=,comm=,args= 2>/dev/null | awk '
        $2 ~ /^python([0-9.]*)?$/ &&
        $0 ~ /(^|[[:space:]])tools\/run_overnight_controller[.]py([[:space:]]|$)/ {print}
    '
}

if lock_is_held; then
    echo "campaign lock is held; refusing activation: $LOCK_FILE" >&2
    exit 1
fi

running_process=$(overnight_controller_processes | head -n 1)
if [ -n "$running_process" ]; then
    echo "overnight Controller process is still running; refusing activation" >&2
    exit 1
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

for relative in "${files[@]}"; do
    source="$STAGE_ROOT/$relative"
    if [ ! -f "$source" ]; then
        echo "staged file is missing: $source" >&2
        exit 1
    fi
done

python3 -m py_compile \
    "$STAGE_ROOT/harness4h3/remote/config.py" \
    "$STAGE_ROOT/harness4h3/remote/comfyui_lease.py" \
    "$STAGE_ROOT/harness4h3/remote/lane_packer.py" \
    "$STAGE_ROOT/harness4h3/remote/pipeline.py" \
    "$STAGE_ROOT/harness4h3/remote/power.py" \
    "$STAGE_ROOT/harness4h3/remote/round_gate.py" \
    "$STAGE_ROOT/harness4h3/remote/importer.py" \
    "$STAGE_ROOT/harness4h3/remote/scheduler.py" \
    "$STAGE_ROOT/harness4h3/remote/ssh.py" \
    "$STAGE_ROOT/harness4h3/benchmark/capabilities.py" \
    "$STAGE_ROOT/harness4h3/controller/context.py" \
    "$STAGE_ROOT/harness4h3/controller/provider.py" \
    "$STAGE_ROOT/harness4h3/controller/schemas.py" \
    "$STAGE_ROOT/harness4h3/controller/round_policy.py" \
    "$STAGE_ROOT/harness4h3/memory/discovery_digest.py" \
    "$STAGE_ROOT/harness4h3/memory/observation.py" \
    "$STAGE_ROOT/research/experiments/remote_h3_closed_loop.py" \
    "$STAGE_ROOT/tools/h3_checkpoint_stage.py" \
    "$STAGE_ROOT/tools/h3_real_train_worker.py" \
    "$STAGE_ROOT/tools/summarize_remote_validation.py"
bash -n \
    "$STAGE_ROOT/tools/activate-remote-pipeline.sh" \
    "$STAGE_ROOT/tools/controller-wait-launch.sh" \
    "$STAGE_ROOT/tools/comfyui-wait-launch.sh" \
    "$STAGE_ROOT/tools/remote-campaign-service.sh" \
    "$STAGE_ROOT/tools/remote-campaign-supervisor.sh" \
    "$STAGE_ROOT/tools/remote-validation-window.sh" \
    "$STAGE_ROOT/tools/remote-idle-autostart.sh" \
    "$STAGE_ROOT/tools/remote-pipeline-handoff.sh"

stop_old_controller_launchers

mkdir -p -- "$BACKUP_ROOT"
for relative in "${files[@]}"; do
    source="$STAGE_ROOT/$relative"
    target="$REPO_ROOT/$relative"
    if [ -f "$target" ]; then
        mkdir -p -- "$BACKUP_ROOT/$(dirname "$relative")"
        cp -p -- "$target" "$BACKUP_ROOT/$relative"
    fi
    temporary="$target.pipeline-v2.tmp.$$"
    cp -- "$source" "$temporary"
    if [[ "$relative" == tools/*.sh ]]; then
        chmod +x -- "$temporary"
    fi
    mv -f -- "$temporary" "$target"
done

python3 -m py_compile \
    "$REPO_ROOT/harness4h3/remote/config.py" \
    "$REPO_ROOT/harness4h3/remote/comfyui_lease.py" \
    "$REPO_ROOT/harness4h3/remote/lane_packer.py" \
    "$REPO_ROOT/harness4h3/remote/pipeline.py" \
    "$REPO_ROOT/harness4h3/remote/power.py" \
    "$REPO_ROOT/harness4h3/remote/round_gate.py" \
    "$REPO_ROOT/harness4h3/remote/importer.py" \
    "$REPO_ROOT/harness4h3/remote/scheduler.py" \
    "$REPO_ROOT/harness4h3/remote/ssh.py" \
    "$REPO_ROOT/harness4h3/benchmark/capabilities.py" \
    "$REPO_ROOT/harness4h3/controller/context.py" \
    "$REPO_ROOT/harness4h3/controller/provider.py" \
    "$REPO_ROOT/harness4h3/controller/schemas.py" \
    "$REPO_ROOT/harness4h3/controller/round_policy.py" \
    "$REPO_ROOT/harness4h3/memory/discovery_digest.py" \
    "$REPO_ROOT/harness4h3/memory/observation.py" \
    "$REPO_ROOT/research/experiments/remote_h3_closed_loop.py" \
    "$REPO_ROOT/tools/h3_checkpoint_stage.py" \
    "$REPO_ROOT/tools/h3_real_train_worker.py" \
    "$REPO_ROOT/tools/summarize_remote_validation.py"
bash -n \
    "$REPO_ROOT/tools/activate-remote-pipeline.sh" \
    "$REPO_ROOT/tools/controller-wait-launch.sh" \
    "$REPO_ROOT/tools/comfyui-wait-launch.sh" \
    "$REPO_ROOT/tools/remote-campaign-service.sh" \
    "$REPO_ROOT/tools/remote-campaign-supervisor.sh" \
    "$REPO_ROOT/tools/remote-validation-window.sh" \
    "$REPO_ROOT/tools/remote-idle-autostart.sh" \
    "$REPO_ROOT/tools/remote-pipeline-handoff.sh"
printf 'activated pipeline v2; backup=%s\n' "$BACKUP_ROOT"

if [ "${REMOTE_START_CONTROLLER:-1}" != "0" ]; then
    mkdir -p -- "$(dirname "$LAUNCHER_LOG")"
    nohup env \
        CONTROLLER_GPU_LEASE_FILE="$CAMPAIGN_ROOT/.controller-gpu-lease.json" \
        CONTROLLER_HANDOFF_HOLD_FILE="$CAMPAIGN_ROOT/.controller-handoff-hold.json" \
        CONTROLLER_WORKER_LEASE_FILE="$CAMPAIGN_ROOT/.h3-worker-gpu-lease.json" \
        CONTROLLER_COMFY_LEASE_FILE="$CAMPAIGN_ROOT/.comfyui-gpu-lease.json" \
        bash "$REPO_ROOT/tools/controller-wait-launch.sh" \
        >> "$LAUNCHER_LOG" 2>&1 < /dev/null &
    printf 'started Controller launcher pid=%s log=%s\n' "$!" "$LAUNCHER_LOG"
fi
