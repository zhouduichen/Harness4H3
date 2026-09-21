#!/usr/bin/env bash
set -u

# Keep the Controller service queued behind other GPU jobs.  A failed vLLM
# startup must return to this loop instead of terminating the queue worker.
MODEL_SOURCE=/data/models/controller-llm/Qwen3.5-35B-A3B-FP8
# /data is NFS on this host.  A completed local cache removes repeated
# multi-minute cold starts when ComfyUI temporarily owns GPU0.  The cache is
# opt-in by readiness marker: an interrupted copy is never served.
MODEL_CACHE=${CONTROLLER_MODEL_CACHE:-/tmp/harness4h3-controller-model/Qwen3.5-35B-A3B-FP8}
MODEL="$MODEL_SOURCE"
if [ -f "$MODEL_CACHE/.ready" ] && [ -f "$MODEL_CACHE/config.json" ]; then
    MODEL="$MODEL_CACHE"
fi
LOG=/data/models/controller-llm/controller-vllm.log
PY=/data/models/envs/sglang_h3/bin/vllm
# Leave a small allowance for nvidia-smi rounding/reservation differences.
# The per-shard guard and dynamic gpu_memory_utilization remain in force.
TOTAL_MIN_FREE_MIB=${CONTROLLER_TOTAL_MIN_FREE_MIB:-39900}
MAX_TENSOR_PARALLEL=${CONTROLLER_MAX_TENSOR_PARALLEL:-4}
ALLOWED_TENSOR_PARALLEL=${CONTROLLER_ALLOWED_TENSOR_PARALLEL:-1,2,4}
COMFY_GPU_INDEX=${CONTROLLER_COMFY_GPU_INDEX:-0}
COMFY_PORT=${CONTROLLER_COMFY_PORT:-8188}
# ComfyUI remains a service process, but it is shareable with the Controller
# only after its queue is empty and its H3 cache has actually been unloaded.
# A small CUDA context is acceptable; a resident model is not.
COMFY_MAX_IDLE_USED_MIB=${CONTROLLER_COMFY_MAX_IDLE_USED_MIB:-2048}
COMFY_LEASE_FILE=${CONTROLLER_COMFY_LEASE_FILE:-/home/intern/huangjiahao/Harness4H3-rsi/work/remote-h3-controller-20260914/.comfyui-gpu-lease.json}
COMFY_LEASE_MAX_AGE_SECONDS=${CONTROLLER_COMFY_LEASE_MAX_AGE_SECONDS:-21600}
# The training scheduler publishes this short-lived marker before launching a
# worker.  The Controller launcher must treat those cards as occupied even if
# the worker has not yet appeared in nvidia-smi (or if vLLM is restarting).
WORKER_LEASE_FILE=${CONTROLLER_WORKER_LEASE_FILE:-/home/intern/huangjiahao/Harness4H3-rsi/work/remote-h3-controller-20260914/.h3-worker-gpu-lease.json}
WORKER_LEASE_MAX_AGE_SECONDS=${CONTROLLER_WORKER_LEASE_MAX_AGE_SECONDS:-86400}
CONTROLLER_LEASE_FILE=${CONTROLLER_GPU_LEASE_FILE:-/home/intern/huangjiahao/Harness4H3-rsi/work/remote-h3-controller-20260914/.controller-gpu-lease.json}
CONTROLLER_LEASE_MAX_AGE_SECONDS=${CONTROLLER_GPU_LEASE_MAX_AGE_SECONDS:-3600}
CONTROLLER_HOLD_FILE=${CONTROLLER_HANDOFF_HOLD_FILE:-/home/intern/huangjiahao/Harness4H3-rsi/work/remote-h3-controller-20260914/.controller-handoff-hold.json}
HANDOFF_HOLD_MAX_AGE_SECONDS=${CONTROLLER_HANDOFF_HOLD_MAX_AGE_SECONDS:-1800}
CONTROLLER_RELEASE_FILE=${CONTROLLER_RELEASE_FILE:-/home/intern/huangjiahao/Harness4H3-rsi/work/remote-h3-controller-20260914/.controller-release.json}
CONTROLLER_PID_FILE=${CONTROLLER_PID_FILE:-/home/intern/huangjiahao/Harness4H3-rsi/work/remote-h3-controller-20260914/.controller-vllm.pid}
RETRY_SECONDS=${CONTROLLER_RETRY_SECONDS:-30}
RACE_RECHECK_SECONDS=${CONTROLLER_RACE_RECHECK_SECONDS:-5}
STABILITY_SAMPLES=${CONTROLLER_STABILITY_SAMPLES:-3}
STABILITY_RECHECK_SECONDS=${CONTROLLER_STABILITY_RECHECK_SECONDS:-5}
# Match the controller client context budget.  This keeps KV-cache reservation
# bounded and leaves more cards available for the training scheduler; callers
# may explicitly raise it when serving a model with a larger verified window.
MAX_MODEL_LEN=${CONTROLLER_MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${CONTROLLER_MAX_NUM_SEQS:-4}
API_PROBE_INTERVAL_SECONDS=${CONTROLLER_API_PROBE_INTERVAL_SECONDS:-15}
# A failed TP launch must yield its cards promptly so the scheduler can try a
# smaller degree.  The remote FP8 checkpoint can spend over five minutes in
# import/weight loading before the API is probeable, especially after a
# handoff, so cover that measured cold-start path without waiting forever.
API_STARTUP_TIMEOUT_SECONDS=${CONTROLLER_API_STARTUP_TIMEOUT_SECONDS:-900}
API_FAILURE_LIMIT=${CONTROLLER_API_FAILURE_LIMIT:-4}
# vLLM may wait for an in-flight HTTP request after SIGTERM.  Do not let a
# graceful handoff hold the launcher (and therefore the GPU lease) forever.
# The campaign only sees the release after this bounded reap completes.
STOP_GRACE_SECONDS=${CONTROLLER_STOP_GRACE_SECONDS:-20}
MAX_LAUNCH_GPU_UTILIZATION=${CONTROLLER_MAX_LAUNCH_GPU_UTILIZATION:-15}
# When an evaluator owns one or more cards, keep enough unblocked cards for a
# distributed successor.  With the normal single ComfyUI evaluator on GPU0
# this reduces the Controller to TP=1; after the lease is released, TP=2/4 is
# again allowed and the Controller can use otherwise idle capacity.
OVERLAP_MIN_TRAINING_GPUS=${CONTROLLER_OVERLAP_MIN_TRAINING_GPUS:-2}
# When ComfyUI owns one card and no worker has actually acquired the remaining
# queue, keep the Controller at TP1 so two cards remain available for the
# distributed speculative worker.  TP2 would leave only one non-ComfyUI card
# and deadlock a worker whose minimum is two GPUs.  Operators may explicitly
# raise this value when they prefer Controller latency over overlap capacity;
# a live worker lease still wins and forces the Controller down to TP1.
EVALUATION_MAX_TENSOR_PARALLEL=${CONTROLLER_EVALUATION_MAX_TENSOR_PARALLEL:-1}
# The detached campaign uses 8001 for the server-local Controller and 8000
# for a watcher-owned fallback.  Detect the former by default so a launcher
# restart cannot create a duplicate vLLM instance on another GPU group.
EXTERNAL_CONTROLLER_PORTS=${CONTROLLER_EXTERNAL_PORTS:-8001}
# A bad tensor-parallel startup can consume the available cards for many
# minutes before vLLM exits.  Cool that parallel degree down after a launch
# that never became healthy, so the queue can fall back to a known-good degree
# (normally TP=1) instead of repeating the same expensive failure all night.
TP_COOLDOWN_SECONDS=${CONTROLLER_TP_COOLDOWN_SECONDS:-180}

# This server has CUDA runtime libraries but no nvcc.  FlashInfer's sampler
# tries to JIT-build a CUDA extension during vLLM warmup; use the native
# sampler and eager execution so Controller startup does not require nvcc.
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p -- "$(dirname "$CONTROLLER_PID_FILE")" "$(dirname "$CONTROLLER_LEASE_FILE")" "$(dirname "$CONTROLLER_HOLD_FILE")"

printf '%s controller launcher started (model=%s source=%s total_min_free_mib=%s max_tensor_parallel=%s allowed_tensor_parallel=%s comfy_gpu=%s)\n' \
    "$(date -Is)" "$MODEL" "$MODEL_SOURCE" "$TOTAL_MIN_FREE_MIB" "$MAX_TENSOR_PARALLEL" "$ALLOWED_TENSOR_PARALLEL" "$COMFY_GPU_INDEX" >> "$LOG"

tp1_cooldown_until=0
tp2_cooldown_until=0
tp4_cooldown_until=0
child_api_ready=0

refresh_cooled_tensor_parallel() {
    local now="$(date +%s)" cooled="" tp until
    for tp in 1 2 4; do
        case "$tp" in
            1) until="$tp1_cooldown_until" ;;
            2) until="$tp2_cooldown_until" ;;
            4) until="$tp4_cooldown_until" ;;
        esac
        if [ "$until" -gt "$now" ]; then
            cooled="${cooled:+$cooled,}$tp"
        fi
    done
    COOLED_TENSOR_PARALLEL="$cooled"
}

cooldown_tensor_parallel() {
    local tp="$1" now until
    now=$(date +%s)
    until=$((now + TP_COOLDOWN_SECONDS))
    case "$tp" in
        1) tp1_cooldown_until="$until" ;;
        2) tp2_cooldown_until="$until" ;;
        4) tp4_cooldown_until="$until" ;;
        *) return 0 ;;
    esac
    printf '%s tensor_parallel=%s startup was unhealthy (rc=%s); cooling it down for %ss\n' \
        "$(date -Is)" "$tp" "${2:-unknown}" "$TP_COOLDOWN_SECONDS" >> "$LOG"
}

child_pid=""
child_gpus=""
ORPHAN_PIDS=""
external_controller_announced=0
child_exit_was_controlled=0

write_controller_lease() {
    local state="$1" owner_pid="$2" gpus="$3" now expires tmp
    now=$(date +%s)
    expires=$((now + CONTROLLER_LEASE_MAX_AGE_SECONDS))
    tmp="${CONTROLLER_LEASE_FILE}.tmp.$$"
    printf '{"allocated_gpus":[%s],"created_at":%s,"expires_at":%s,"owner_pid":%s,"state":"%s"}\n' \
        "$gpus" "$now" "$expires" "$owner_pid" "$state" > "$tmp"
    mv -f -- "$tmp" "$CONTROLLER_LEASE_FILE"
}

clear_controller_lease() {
    # Only the launcher writes this exact campaign-scoped marker.  The caller
    # invokes this after the owned child has exited, so no unrelated process
    # can be made invisible to the scheduler.
    rm -f -- "$CONTROLLER_LEASE_FILE"
}

controller_lease_owner_is() {
    local expected_pid="$1"
    [ -f "$CONTROLLER_LEASE_FILE" ] || return 1
    python3 - "$CONTROLLER_LEASE_FILE" "$expected_pid" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        value = json.load(handle)
    raise SystemExit(0 if int(value.get("owner_pid", 0)) == int(sys.argv[2]) else 1)
except (OSError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

clear_controller_lease_if_owner() {
    # An old D-state orphan may outlive a replacement Controller.  Never let
    # reaping that old PID delete the replacement child's newer lease.
    if controller_lease_owner_is "$1"; then
        clear_controller_lease
    fi
}

handoff_hold_active() {
    local mtime now age
    [ -f "$CONTROLLER_HOLD_FILE" ] || return 1
    mtime=$(stat -c %Y -- "$CONTROLLER_HOLD_FILE" 2>/dev/null || true)
    case "$mtime" in
        ''|*[!0-9]*) return 0 ;;
    esac
    now=$(date +%s)
    age=$((now - mtime))
    if [ "$age" -ge 0 ] && [ "$age" -gt "$HANDOFF_HOLD_MAX_AGE_SECONDS" ]; then
        # A campaign crash must not leave the watcher permanently paused. The
        # exact marker is campaign-scoped and has already exceeded its lease.
        rm -f -- "$CONTROLLER_HOLD_FILE"
        printf '%s stale Controller handoff hold expired after %ss\n' \
            "$(date -Is)" "$age" >> "$LOG"
        return 1
    fi
    return 0
}

kill_process_tree() {
    local root="$1" node
    for node in $(pgrep -P "$root" 2>/dev/null || true); do
        kill_process_tree "$node"
    done
    kill -TERM "$root" 2>/dev/null || true
}

kill_process_tree_hard() {
    local root="$1" node
    for node in $(pgrep -P "$root" 2>/dev/null || true); do
        kill_process_tree_hard "$node"
    done
    kill -KILL "$root" 2>/dev/null || true
}

stop_child() {
    terminate_owned_child
    if ! child_is_alive; then
        clear_controller_lease
    fi
    rm -f -- "$CONTROLLER_PID_FILE"
    exit 143
}

comfy_lease_file_for_gpu() {
    local gpu="$1"
    if [ "$gpu" -eq "$COMFY_GPU_INDEX" ]; then
        printf '%s\n' "$COMFY_LEASE_FILE"
    else
        printf '%s-%s.json\n' "${COMFY_LEASE_FILE%.json}" "$gpu"
    fi
}

comfy_benchmark_lease_active() {
    local gpu="${1:-$COMFY_GPU_INDEX}" file mtime now age
    file=$(comfy_lease_file_for_gpu "$gpu")
    if [ ! -f "$file" ]; then
        return 1
    fi
    # A crashed campaign must not strand an evaluator GPU forever.  The lease
    # is renewed at every benchmark boundary; a bounded age is a safe fallback
    # for an SSH-launched campaign whose PID is not visible on this host.
    mtime=$(stat -c %Y -- "$file" 2>/dev/null || true)
    case "$mtime" in
        ''|*[!0-9]*) return 0 ;;
    esac
    now=$(date +%s)
    age=$((now - mtime))
    if [ "$age" -lt 0 ] || [ "$age" -le "$COMFY_LEASE_MAX_AGE_SECONDS" ]; then
        return 0
    fi
    return 1
}

selected_gpu_has_comfy_lease() {
    local gpu
    IFS=',' read -r -a gpu_list <<< "$child_gpus"
    for gpu in "${gpu_list[@]}"; do
        if comfy_benchmark_lease_active "$gpu"; then
            return 0
        fi
    done
    return 1
}
trap stop_child TERM INT

child_is_alive() {
    if [ -z "$child_pid" ] || ! kill -0 "$child_pid" 2>/dev/null; then
        return 1
    fi
    case "$(ps -o stat= -p "$child_pid" 2>/dev/null | tr -d '[:space:]')" in
        Z*) return 1 ;;
    esac
    return 0
}

terminate_owned_child() {
    local deadline
    if ! child_is_alive; then
        wait "$child_pid" 2>/dev/null || true
        return 0
    fi
    kill_process_tree "$child_pid"
    deadline=$((SECONDS + STOP_GRACE_SECONDS))
    while child_is_alive && [ "$SECONDS" -lt "$deadline" ]; do
        sleep 1
    done
    if child_is_alive; then
        printf '%s owned vLLM child did not exit after %ss; force reaping process tree\n' "$(date -Is)" "$STOP_GRACE_SECONDS" >> "$LOG"
        kill_process_tree_hard "$child_pid"
    fi
    if child_is_alive; then
        # A process blocked in uninterruptible NFS/RPC I/O can survive KILL.
        # Bash wait would then block this launcher indefinitely.  Keep the
        # exact PID in a small orphan queue and poll it without starting a
        # replacement Controller on potentially conflicting resources.
        case ",$ORPHAN_PIDS," in
            *",$child_pid,"*) ;;
            *) ORPHAN_PIDS="${ORPHAN_PIDS:+$ORPHAN_PIDS,}$child_pid" ;;
        esac
        printf '%s owned vLLM child pid=%s remains uninterruptible; polling exact orphan without wait\n' \
            "$(date -Is)" "$child_pid" >> "$LOG"
        return 124
    fi
    # Reap only after the exact child is no longer alive.  Never call wait on
    # a D-state process: that was the source of the launcher deadlock.
    wait "$child_pid" 2>/dev/null || true
}

reap_orphans() {
    local remaining="" pid
    [ -n "$ORPHAN_PIDS" ] || return 0
    IFS=',' read -r -a orphan_list <<< "$ORPHAN_PIDS"
    for pid in "${orphan_list[@]}"; do
        case "$pid" in
            ''|*[!0-9]*) continue ;;
        esac
        if kill -0 "$pid" 2>/dev/null; then
            remaining="${remaining:+$remaining,}$pid"
            continue
        fi
        # The PID belongs to this launcher.  It is now reapable and this wait
        # returns immediately, including when the child became a zombie.
        wait "$pid" 2>/dev/null || true
        # No replacement child can be launched while ORPHAN_PIDS is non-empty.
        # Removing the exact Controller lease here makes the reservation
        # lifetime end with the owned orphan rather than with its TTL.
        clear_controller_lease_if_owner "$pid"
        printf '%s exact orphan pid=%s was reaped; GPU queue is open\n' "$(date -Is)" "$pid" >> "$LOG"
    done
    ORPHAN_PIDS="$remaining"
}

orphan_gpu_quiescent() {
    local pid
    [ -n "$ORPHAN_PIDS" ] || return 0
    # KILL cannot remove a process blocked in NFS/RPC D-state.  It is safe to
    # reopen the Controller queue only after nvidia-smi confirms that each
    # exact orphan owns no CUDA compute context; the orphan remains tracked and
    # is reaped later without a blocking wait.
    IFS=',' read -r -a orphan_list <<< "$ORPHAN_PIDS"
    for pid in "${orphan_list[@]}"; do
        case "$pid" in
            ''|*[!0-9]*) continue ;;
        esac
        if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | awk -F, -v target_pid="$pid" '
                {
                    value=$1
                    gsub(/[[:space:]]/, "", value)
                    if (value == target_pid) { found=1; exit }
                }
                END { exit(found ? 0 : 1) }'; then
            return 1
        fi
    done
    return 0
}

controller_api_ready() {
    curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null \
        | grep -q 'qwen3.5-controller'
}

external_controller_api_ready() {
    local port
    [ -n "$EXTERNAL_CONTROLLER_PORTS" ] || return 1
    IFS=',' read -r -a external_ports <<< "$EXTERNAL_CONTROLLER_PORTS"
    for port in "${external_ports[@]}"; do
        case "$port" in
            ''|*[!0-9]*) continue ;;
        esac
        if curl -fsS --max-time 5 "http://127.0.0.1:$port/v1/models" 2>/dev/null \
            | grep -Fq 'qwen3.5-controller'; then
            EXTERNAL_CONTROLLER_SELECTED_PORT="$port"
            return 0
        fi
    done
    return 1
}

monitor_child() {
    local started_at now failures=0 api_ready=0 last_probe=0
    child_api_ready=0
    child_exit_was_controlled=0
    started_at=$(date +%s)
    while child_is_alive; do
        if handoff_hold_active; then
            printf '%s campaign holds Controller lane for worker handoff; stopping owned vLLM child\n' \
                "$(date -Is)" >> "$LOG"
            child_exit_was_controlled=1
            terminate_owned_child
            break
        fi
        if [ -f "$CONTROLLER_RELEASE_FILE" ]; then
            printf '%s campaign requested Controller release; stopping owned vLLM child\n' \
                "$(date -Is)" >> "$LOG"
            child_exit_was_controlled=1
            terminate_owned_child
            # The marker is a one-shot handoff signal.  Only remove this
            # exact configured file; do not touch another campaign's state.
            rm -f -- "$CONTROLLER_RELEASE_FILE"
            break
        fi
        if comfy_benchmark_overlap_active && [ ! -f "$WORKER_LEASE_FILE" ] \
            && [ "$tensor_parallel" -gt "$EVALUATION_MAX_TENSOR_PARALLEL" ]; then
            printf '%s evaluation overlap requires Controller TP<=%s; restarting owned TP%s child to free worker lane\n' \
                "$(date -Is)" "$EVALUATION_MAX_TENSOR_PARALLEL" "$tensor_parallel" >> "$LOG"
            child_exit_was_controlled=1
            terminate_owned_child
            break
        fi
        if [ -f "$WORKER_LEASE_FILE" ] && [ "$tensor_parallel" -gt 1 ]; then
            printf '%s worker lease is active; restarting owned TP%s child so Controller returns to TP1\n' \
                "$(date -Is)" "$tensor_parallel" >> "$LOG"
            child_exit_was_controlled=1
            terminate_owned_child
            break
        fi
        if selected_gpu_has_comfy_lease; then
            printf '%s ComfyUI owns a Controller shard GPU (%s); stopping Controller child before evaluation\n' \
                "$(date -Is)" "$child_gpus" >> "$LOG"
            child_exit_was_controlled=1
            terminate_owned_child
            break
        fi
        if [[ ",$child_gpus," == *",$COMFY_GPU_INDEX,"* ]] && comfy_process_active; then
            if ! comfy_queue_idle; then
                printf '%s ComfyUI queue is active on GPU%s; stopping Controller child before evaluation\n' \
                    "$(date -Is)" "$COMFY_GPU_INDEX" >> "$LOG"
                child_exit_was_controlled=1
                terminate_owned_child
                break
            fi
            if ! comfy_cache_released; then
                printf '%s shared primary ComfyUI became active on GPU%s; stopping Controller child before cache reload\n' \
                    "$(date -Is)" "$COMFY_GPU_INDEX" >> "$LOG"
                child_exit_was_controlled=1
                terminate_owned_child
                break
            fi
        fi
        now=$(date +%s)
        if [ $((now - last_probe)) -ge "$API_PROBE_INTERVAL_SECONDS" ]; then
            last_probe="$now"
        elif [ "$api_ready" -eq 1 ]; then
            sleep 2
            continue
        fi
        if controller_api_ready; then
            if [ "$api_ready" -eq 0 ]; then
                printf '%s Controller API ready (max_model_len=%s max_num_seqs=%s)\n' \
                    "$(date -Is)" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" >> "$LOG"
            fi
            api_ready=1
            child_api_ready=1
            failures=0
        else
            now=$(date +%s)
            if [ "$api_ready" -eq 1 ] || [ $((now - started_at)) -ge "$API_STARTUP_TIMEOUT_SECONDS" ]; then
                failures=$((failures + 1))
                printf '%s Controller API probe failed (%s/%s)\n' \
                    "$(date -Is)" "$failures" "$API_FAILURE_LIMIT" >> "$LOG"
                if [ "$failures" -ge "$API_FAILURE_LIMIT" ]; then
                    printf '%s Controller API unhealthy; terminating vLLM child\n' \
                        "$(date -Is)" >> "$LOG"
                    terminate_owned_child
                    break
                fi
            fi
        fi
        sleep 2
    done
    if child_is_alive; then
        # ``terminate_owned_child`` already recorded the exact PID.  Returning
        # here is essential: waiting on a D-state child can freeze the queue.
        return 124
    fi
    wait "$child_pid" 2>/dev/null
    return $?
}

# Return the largest currently feasible GPU group as
#   tensor_parallel_size|CUDA_VISIBLE_DEVICES|free_sum|free_min
# The total threshold is calibrated from the failed TP=1 startup. Requiring
# each shard to have at least total/tp free avoids selecting an imbalanced set.
# The default TP values are the divisors of Qwen3.5's 16 attention heads; this
# prevents the queue from repeatedly launching an invalid TP=3 configuration.
# Prefer the largest feasible TP so an idle Controller uses otherwise-unused
# cards.  The worker/evaluator leases are already excluded by blocked_gpu, so
# this cannot steal cards from an active training or benchmark reservation.
comfy_process_active() {
    [ -n "$(comfy_process_pid)" ]
}

comfy_process_pid() {
    # ComfyUI is launched with CUDA_VISIBLE_DEVICES, so its command line does
    # not contain a --cuda-device argument.  The campaign maps one API port
    # to one GPU; use that stable identity when checking a pre-existing daemon.
    pgrep -f "main.py.*--port[ =]$COMFY_PORT" 2>/dev/null | head -n 1
}

comfy_queue_idle() {
    local payload
    payload=$(curl -fsS --max-time 2 "http://127.0.0.1:$COMFY_PORT/queue" 2>/dev/null | tr -d '[:space:]' || true)
    if [ -z "$payload" ]; then
        # An unavailable API is not proof that the evaluator is idle.
        return 1
    fi
    case "$payload" in
        *'"queue_running":[]'*'"queue_pending":[]'*) return 0 ;;
        *) return 1 ;;
    esac
}

comfy_cache_released() {
    local pid used
    pid=$(comfy_process_pid)
    if [ -z "$pid" ]; then
        return 0
    fi
    if ! comfy_queue_idle; then
        return 1
    fi
    # Inspect only the ComfyUI process.  Other users' allocations on the same
    # card are independent scheduler inputs and must not look like a resident
    # ComfyUI model.
    used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, -v target_pid="$pid" '
            {
                pid=$1; used=$2
                gsub(/[[:space:]]/, "", pid)
                gsub(/[[:space:]]/, "", used)
                if (pid == target_pid) { print used; exit }
            }')
    if [ -z "$used" ]; then
        # An idle ComfyUI process with no compute entry has no model resident.
        return 0
    fi
    case "$used" in
        ''|*[!0-9]*) return 1 ;;
    esac
    [ "$used" -le "$COMFY_MAX_IDLE_USED_MIB" ]
}

comfy_idle_compute_pid_allowed() {
    local gpu="$1" compute_pids="$2" comfy_pid pid
    [ "$gpu" -eq "$COMFY_GPU_INDEX" ] || return 1
    comfy_process_active || return 1
    comfy_cache_released || return 1
    comfy_pid=$(comfy_process_pid)
    [ -n "$comfy_pid" ] || return 1
    # An idle primary ComfyUI context is the only compute PID that may share
    # GPU0. Any foreign PID, even at 0% utilization, keeps the candidate
    # fail-closed because utilization is not an ownership proof.
    for pid in $compute_pids; do
        [ "$pid" = "$comfy_pid" ] || return 1
    done
    return 0
}

comfy_benchmark_overlap_active() {
    local gpu
    for gpu in 0 1 2 3; do
        if comfy_benchmark_lease_active "$gpu"; then
            return 0
        fi
    done
    return 1
}

select_candidate() {
    refresh_cooled_tensor_parallel
    local effective_max_tp="$MAX_TENSOR_PARALLEL"
    local overlap_capacity_limited=0
    local preferred_gpu=""
    if comfy_process_active && comfy_cache_released; then
        # When the resident primary ComfyUI process is genuinely idle, prefer
        # sharing its card over a tiny free-memory advantage on another card.
        # candidate_is_quiet still verifies that this is the only compute PID
        # before launch, so a foreign job cannot be displaced by this hint.
        preferred_gpu="$COMFY_GPU_INDEX"
    fi
    if [ "$OVERLAP_MIN_TRAINING_GPUS" -gt 0 ] && comfy_benchmark_overlap_active && [ -n "$BLOCKED_GPUS" ]; then
        local usable_gpu_count=4 blocked_gpu seen_blocked= blocked
        IFS=',' read -r -a blocked_list <<< "$BLOCKED_GPUS"
        for blocked in "${blocked_list[@]}"; do
            case ",$seen_blocked," in
                *",$blocked,"*) continue ;;
            esac
            case "$blocked" in
                0|1|2|3)
                    seen_blocked="${seen_blocked:+$seen_blocked,}$blocked"
                    usable_gpu_count=$((usable_gpu_count - 1))
                    ;;
            esac
        done
        local overlap_max_tp=$((usable_gpu_count - OVERLAP_MIN_TRAINING_GPUS))
        if [ "$overlap_max_tp" -lt "$effective_max_tp" ]; then
            effective_max_tp="$overlap_max_tp"
        fi
        overlap_capacity_limited=1
    fi
    # During an evaluation window, cap the Controller before candidate search
    # so the remaining cards can satisfy the distributed worker minimum. A
    # live worker lease remains authoritative and keeps the Controller at the
    # same or smaller degree so it cannot steal the worker's cards.
    if [ "$overlap_capacity_limited" -eq 1 ] \
        && [ ! -f "$WORKER_LEASE_FILE" ]; then
        if [ "$effective_max_tp" -gt "$EVALUATION_MAX_TENSOR_PARALLEL" ]; then
            effective_max_tp="$EVALUATION_MAX_TENSOR_PARALLEL"
            printf '%s evaluation has no live worker lease; capping Controller TP at %s to preserve two-card worker overlap\n' \
                "$(date -Is)" "$effective_max_tp" >> "$LOG"
        fi
    fi
    nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null \
        | awk -F, -v total="$TOTAL_MIN_FREE_MIB" -v max_tp="$effective_max_tp" -v allowed_tp="$ALLOWED_TENSOR_PARALLEL" -v cooled_tp="$COOLED_TENSOR_PARALLEL" -v blocked_gpu="$BLOCKED_GPUS" -v preferred_gpu="$preferred_gpu" '
            function visit(start, depth, sum, min_free, min_ratio, selected, i, req, shard_util) {
                if (depth > 0) {
                    req = total / depth
                    if (index("," allowed_tp ",", "," depth ",") > 0 && index("," cooled_tp ",", "," depth ",") == 0 && sum >= total && min_free >= req) {
                        shard_util = (min_ratio - 0.01 < 0.90 ? min_ratio - 0.01 : 0.90)
                        preferred = preferred_gpu != "" && index("," selected ",", "," preferred_gpu ",") > 0
                        best_preferred = preferred_gpu != "" && index("," best_selected ",", "," preferred_gpu ",") > 0
                        if (shard_util > 0 && (best_depth == 0 || depth > best_depth || (depth == best_depth && (preferred && !best_preferred || preferred == best_preferred && min_free > best_min)))) {
                            best_depth = depth
                            best_sum = sum
                            best_min = min_free
                            best_util = shard_util
                            best_selected = selected
                        }
                    }
                }
                if (depth >= max_tp) return
                for (i = start; i <= count; i++) {
                    visit(i + 1, depth + 1, sum + free_mem[i],
                          depth == 0 ? free_mem[i] : (min_free < free_mem[i] ? min_free : free_mem[i]),
                          depth == 0 ? free_mem[i] / total_mem[i] : (min_ratio < free_mem[i] / total_mem[i] ? min_ratio : free_mem[i] / total_mem[i]),
                          selected == "" ? gpu[i] : selected "," gpu[i])
                }
            }
            {
                gpu_id = $1
                gsub(/[[:space:]]/, "", gpu_id)
                if (blocked_gpu != "" && index("," blocked_gpu ",", "," gpu_id ",") > 0) next
                gpu[++count] = gpu_id
                gsub(/[[:space:]]/, "", gpu[count])
                free_mem[count] = $2 + 0
                total_mem[count] = $3 + 0
            }
            END {
                best_depth = 0
                visit(1, 0, 0, 0, 0, "")
                if (best_depth > 0) print best_depth "|" best_selected "|" best_sum "|" best_min "|" best_util
            }'
}

candidate_is_quiet() {
    local selected_gpus="$1" gpu utilization compute_pids
    IFS=',' read -r -a gpu_list <<< "$selected_gpus"
    for gpu in "${gpu_list[@]}"; do
        # A sparse or short-lived CUDA job can report 0% at the exact
        # utilization sample while still owning the card.  Treat any compute
        # process as occupied so the queue can fall back to a smaller
        # disjoint GPU group instead of repeatedly selecting a busy TP4 set.
        compute_pids=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | awk '$1 ~ /^[0-9]+$/ {print $1}')
        if [ -n "$compute_pids" ] && ! comfy_idle_compute_pid_allowed "$gpu" "$compute_pids"; then
            return 1
        fi
        utilization=$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
        case "$utilization" in
            ''|*[!0-9]*) return 1 ;;
        esac
        if [ "$utilization" -gt "$MAX_LAUNCH_GPU_UTILIZATION" ]; then
            return 1
        fi
    done
    return 0
}

candidate_busy_gpus() {
    local selected_gpus="$1" gpu utilization compute_pids busy=""
    IFS=',' read -r -a gpu_list <<< "$selected_gpus"
    for gpu in "${gpu_list[@]}"; do
        compute_pids=$(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | awk '$1 ~ /^[0-9]+$/ {print $1}')
        utilization=$(nvidia-smi --id="$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
        if [ -n "$compute_pids" ] || {
            case "$utilization" in
                ''|*[!0-9]*) true ;;
                *) [ "$utilization" -gt "$MAX_LAUNCH_GPU_UTILIZATION" ] ;;
            esac
        }; then
            busy="${busy:+$busy,}$gpu"
        fi
    done
    printf '%s\n' "$busy"
}

select_quiet_candidate() {
    local candidate tensor_parallel gpus free_sum free_min gpu_memory_utilization busy gpu
    # select_candidate optimizes for the largest TP degree.  If that group is
    # occupied, progressively block only its busy cards and retry so a free
    # TP1/TP2 subset (for example GPU3 during an external 0/1/2 job) can run.
    for _ in 1 2 3 4; do
        candidate=$(select_candidate)
        [ -n "$candidate" ] || return 1
        IFS='|' read -r tensor_parallel gpus free_sum free_min gpu_memory_utilization <<EOF
$candidate
EOF
        if candidate_is_quiet "$gpus"; then
            printf '%s\n' "$candidate"
            return 0
        fi
        busy=$(candidate_busy_gpus "$gpus")
        [ -n "$busy" ] || return 1
        IFS=',' read -r -a busy_list <<< "$busy"
        for gpu in "${busy_list[@]}"; do
            append_blocked_gpu "$gpu"
        done
    done
    return 1
}

refresh_blocked_gpu() {
    # Do not permanently blacklist GPU0.  The campaign's lease calls
    # ComfyUI /free before training/controller allocation.  Once the queue is
    # empty and the process has only a small CUDA context left, live VRAM
    # selection may reuse the card.  Any failed/ambiguous probe stays
    # fail-closed and blocks GPU0.
    BLOCKED_GPUS=""
    append_blocked_gpu() {
        local value="$1"
        [ -n "$value" ] || return 0
        if [ -z "$BLOCKED_GPUS" ]; then
            BLOCKED_GPUS="$value"
        else
            BLOCKED_GPUS="$BLOCKED_GPUS,$value"
        fi
    }
    for comfy_gpu in 0 1 2 3; do
        if comfy_benchmark_lease_active "$comfy_gpu"; then
            append_blocked_gpu "$comfy_gpu"
        elif [ "$comfy_gpu" -eq "$COMFY_GPU_INDEX" ] && comfy_process_active && ! comfy_cache_released; then
            append_blocked_gpu "$comfy_gpu"
        fi
    done
    if [ -f "$WORKER_LEASE_FILE" ]; then
        # Parse only a live, well-formed lease.  An expired marker is ignored
        # so a crashed campaign cannot strand GPUs indefinitely.  A malformed
        # marker is fail-closed by blocking every listed integer it contains.
        worker_gpus=$(python3 - "$WORKER_LEASE_FILE" "$WORKER_LEASE_MAX_AGE_SECONDS" <<'PY'
import json
import os
import sys
import time

try:
    path = sys.argv[1]
    max_age = float(sys.argv[2])
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    now = time.time()
    created = float(value.get("created_at", 0.0))
    expires = float(value.get("expires_at", 0.0))
    if created <= 0 or expires <= 0 or now > expires or now - created > max_age:
        raise SystemExit(0)
    # The campaign PID is the lease owner.  A supervisor crash can leave a
    # fresh-looking 24-hour lease behind even though no worker exists; do not
    # strand all GPUs until TTL expiry.  An owner that is still alive remains
    # fail-closed and continues to block this queue.
    owner_pid = int(value.get("owner_pid", 0))
    if owner_pid > 0:
        try:
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            raise SystemExit(0)
        except PermissionError:
            pass
    values = value.get("allocated_gpus", [])
    if not isinstance(values, list):
        raise SystemExit(0)
    result = []
    for item in values:
        index = int(item)
        if 0 <= index < 128:
            result.append(str(index))
    print(",".join(result))
except Exception:
    # Do not guess on malformed JSON.  The live nvidia-smi process check still
    # protects cards once the worker has actually initialized.
    raise SystemExit(0)
PY
        )
        if [ -n "$worker_gpus" ]; then
            IFS=',' read -r -a worker_gpu_list <<< "$worker_gpus"
            for worker_gpu in "${worker_gpu_list[@]}"; do
                append_blocked_gpu "$worker_gpu"
            done
        fi
    fi
}

# vLLM loads the checkpoint before it performs its first CUDA memory check.
# A single free-memory sample is therefore not a reservation: another job can
# start during weight loading and make the launch fail.  Require the same GPU
# group to remain feasible across several samples immediately before launch.
candidate_is_stable() {
    local expected_tensor_parallel="$1"
    local expected_gpus="$2"
    local sample=0
    local refreshed sample_tensor_parallel sample_gpus
    while [ "$sample" -lt "$STABILITY_SAMPLES" ]; do
        sleep "$STABILITY_RECHECK_SECONDS"
        refresh_blocked_gpu
        refreshed=$(select_quiet_candidate)
        if [ -z "$refreshed" ]; then
            return 1
        fi
        IFS='|' read -r sample_tensor_parallel sample_gpus _ _ _ <<EOF
$refreshed
EOF
        if [ "$sample_tensor_parallel" != "$expected_tensor_parallel" ] || [ "$sample_gpus" != "$expected_gpus" ]; then
            return 1
        fi
        sample=$((sample + 1))
    done
    return 0
}

while true; do
    reap_orphans
    if [ -n "$ORPHAN_PIDS" ] && ! orphan_gpu_quiescent; then
        sleep "$RETRY_SECONDS"
        continue
    fi
    if [ -n "$ORPHAN_PIDS" ]; then
        printf '%s D-state Controller orphan has no CUDA context; reopening GPU queue without wait (pids=%s)\n' \
            "$(date -Is)" "$ORPHAN_PIDS" >> "$LOG"
    fi
    if handoff_hold_active; then
        sleep "$RETRY_SECONDS"
        continue
    fi
    if external_controller_api_ready; then
        if [ "$external_controller_announced" -eq 0 ]; then
            printf '%s reusing healthy external Controller on loopback port %s; own vLLM launch is paused\n' \
                "$(date -Is)" "$EXTERNAL_CONTROLLER_SELECTED_PORT" >> "$LOG"
            external_controller_announced=1
        fi
        sleep "$RETRY_SECONDS"
        continue
    fi
    external_controller_announced=0
    refresh_blocked_gpu
    candidate=$(select_quiet_candidate)
    if [ -z "$candidate" ]; then
        sleep "$RETRY_SECONDS"
        continue
    fi

    IFS='|' read -r tensor_parallel gpus free_sum free_min gpu_memory_utilization <<EOF
$candidate
EOF

    # Re-select immediately before launching so a competing job cannot win
    # the race between GPU selection and vLLM CUDA initialization.
    sleep "$RACE_RECHECK_SECONDS"
    refresh_blocked_gpu
    refreshed=$(select_quiet_candidate)
    if [ -z "$refreshed" ]; then
        printf '%s GPU group %s lost queue reservation; retrying\n' \
            "$(date -Is)" "$gpus" >> "$LOG"
        sleep "$RETRY_SECONDS"
        continue
    fi
    IFS='|' read -r tensor_parallel gpus free_sum free_min gpu_memory_utilization <<EOF
$refreshed
EOF

    if ! candidate_is_stable "$tensor_parallel" "$gpus"; then
        printf '%s GPU group %s was not stable for %s samples; returning to queue\n' \
            "$(date -Is)" "$gpus" "$STABILITY_SAMPLES" >> "$LOG"
        sleep "$RETRY_SECONDS"
        continue
    fi

    printf '%s selecting GPU group %s (tensor_parallel=%s free_sum=%s MiB free_min=%s MiB gpu_memory_utilization=%s)\n' \
        "$(date -Is)" "$gpus" "$tensor_parallel" "$free_sum" "$free_min" "$gpu_memory_utilization" >> "$LOG"
    # Publish the reservation before CUDA initialization.  The scheduler can
    # therefore allocate disjoint worker GPUs without racing vLLM weight load.
    write_controller_lease "launching" "$$" "$gpus"
    export CUDA_VISIBLE_DEVICES="$gpus"
    child_gpus="$gpus"
    "$PY" serve "$MODEL" \
        --host 127.0.0.1 \
        --port 8000 \
        --served-model-name qwen3.5-controller \
        --tensor-parallel-size "$tensor_parallel" \
        --gpu-memory-utilization "$gpu_memory_utilization" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --enforce-eager \
        --language-model-only \
        --trust-remote-code \
        >> "$LOG" 2>&1 &
    child_pid=$!
    write_controller_lease "allocated_for_controller" "$child_pid" "$gpus"
    printf '%s\n' "$child_pid" > "$CONTROLLER_PID_FILE"
    monitor_child
    rc=$?
    if [ -f "$CONTROLLER_PID_FILE" ] && [ "$(tr -d '[:space:]' < "$CONTROLLER_PID_FILE" 2>/dev/null || true)" = "$child_pid" ]; then
        rm -f -- "$CONTROLLER_PID_FILE"
    fi
    if ! child_is_alive; then
        clear_controller_lease
    fi
    child_pid=""
    child_gpus=""
    unset CUDA_VISIBLE_DEVICES
    if [ "$child_api_ready" -eq 0 ] && [ "$child_exit_was_controlled" -eq 0 ]; then
        cooldown_tensor_parallel "$tensor_parallel" "$rc"
    elif [ "$child_api_ready" -eq 0 ]; then
        printf '%s vLLM child was released by a controlled handoff; skipping tensor_parallel cooldown\n' \
            "$(date -Is)" >> "$LOG"
    fi
    printf '%s vLLM exited rc=%s; returning to GPU queue\n' \
        "$(date -Is)" "$rc" >> "$LOG"
    sleep "$RETRY_SECONDS"
done
