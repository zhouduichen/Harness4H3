# ComfyUI GPU Lease and Idle Cache Release Design

**Date:** 2026-09-15

## Goal

让远程闭环在 ComfyUI 没有评测任务时归还 GPU0。ComfyUI 服务进程继续运行，但其模型缓存只在需要时占用显存；训练或其他受调度器管理的实验可以在安全边界申请 GPU0。评测任务开始前，系统重新建立 GPU0 租约，ComfyUI 按需加载模型。

## Ownership boundary

Campaign resource orchestration owns the GPU0 lease. ComfyUI owns model loading and unloading through its existing HTTP API. The scheduler only allocates GPUs after the lease state has been released and a fresh `nvidia-smi` snapshot passes the configured memory waterline. The Controller may choose training resources, but it cannot directly release caches, terminate processes, or change evaluator results.

## Current gap

The scheduler initially reserves GPU0 forever, while the campaign only observes whether the ComfyUI queue is active. A completed queue can therefore leave a large H3 model resident on GPU0 and keep the card unavailable even when no benchmark is running. Existing per-task cache-release policies are runtime/evaluator policies and do not provide a campaign-level resource lease.

## Chosen approach

Use an explicit, fail-closed ComfyUI lease with two safe transitions:

1. `prepare_for_benchmark`: reserve GPU0 before preflight and benchmark submission.
2. `release_if_idle`: after all benchmark artifacts and evidence are persisted, verify that ComfyUI has no running or pending queue item, call `POST /free`, verify the response, and remove GPU0 from scheduler reservations.

The release operation is invoked at the campaign boundary, never from inside an active benchmark task. If queue state, `/free`, or post-release verification is unavailable, GPU0 remains reserved. The campaign continues using the other GPUs when possible; it does not claim that GPU0 was released.

The default policy is `idle_release` at the end of a completed benchmark phase. A benchmark recipe can opt into `warm_cache` for controlled latency comparisons, or `cold_cache` when cold-start latency is explicitly part of the recipe. The cache mode is recorded in the benchmark provenance so results are not compared across unstated cache states.

## Lease state machine

```text
reserved_for_benchmark
        |
        | benchmark queue empty + evidence persisted
        v
release_requested -- failure --> reserved_for_benchmark
        |
        | /free acknowledged + post-check passed
        v
released_for_other_work
        |
        | next benchmark preflight
        v
reserved_for_benchmark
```

The initial state remains `reserved_for_benchmark` for backward compatibility. A released lease is not treated as proof that ComfyUI is available for inference until the next `prepare_for_benchmark` boundary.

## Configuration contract

Add campaign benchmark settings equivalent to:

```yaml
benchmark:
  comfyui_cache_policy: idle_release  # idle_release | warm_cache | cold_cache
```

`idle_release` is the default for remote resource utilization. `warm_cache` preserves the current resident model after evaluation. `cold_cache` releases before each controlled benchmark and keeps the GPU reserved during the benchmark phase. Unsupported values fail configuration validation rather than silently changing measurement semantics.

## Runtime flow

1. Before benchmark preflight, set the scheduler reservation for GPU0 and record the lease state.
2. Run the independent benchmark and evaluator without changing the lease during a task.
3. Persist benchmark artifacts, `EvaluationRecord`, and campaign events.
4. For `idle_release` or `cold_cache`, query `/queue`; release only when both running and pending lists are empty.
5. Call `/free` with model and memory release enabled.
6. Record the API response and a post-release GPU memory snapshot. The post-check must meet the configured low-memory watermark; remove GPU0 from scheduler reservations only after both checks succeed.
7. On the next benchmark, reserve GPU0 before preflight. Normal ComfyUI workflow submission reloads the required checkpoint if it is no longer resident.

Training allocation is still dynamic and occurs only between tasks. A training request never shares GPU0 with a resident ComfyUI model. If GPU0 has not been released, the scheduler treats it as reserved/busy and allocates another valid group or waits according to the Controller plan.

## Observability

Append structured events to `controller-events.jsonl`:

- `comfyui_lease_reserved`: benchmark boundary, cache policy, and reserved indices;
- `comfyui_cache_release_requested`: queue snapshot and reason;
- `comfyui_cache_released`: `/free` response summary, elapsed time, post-release memory snapshot, and released indices;
- `comfyui_cache_release_failed`: safe failure reason and retained reservation.

Events must not contain credentials, raw SSH command secrets, or full unbounded API payloads. Benchmark provenance includes the effective cache policy and whether the lease was released successfully.

## Failure and safety handling

- Never call `/free` while `queue_running` or `queue_pending` is non-empty.
- Queue/API timeout, malformed response, `/free` error, or insufficient post-release memory evidence keeps GPU0 reserved. The memory watermark is the same scheduler safety threshold used for allocation, so a partially unloaded model cannot be mistaken for an available card.
- Release is idempotent: an already-empty cache is a successful released state when the post-check confirms the memory watermark.
- A release failure cannot cancel a completed benchmark, invalidate its evidence, or cause the scheduler to allocate GPU0 optimistically.
- The LLM service is unaffected; its launcher may use 1, 2, or 4 cards selected from live free capacity. This lease does not move, reload, or co-locate the Controller model.

## Testing and acceptance

Unit tests cover queue-active refusal, idle release success, API failure fail-closed behavior, scheduler reservation changes, idempotent release, configuration validation, and event/provenance fields. Integration tests cover benchmark completion followed by release and the next benchmark re-reserving GPU0.

Remote acceptance requires:

1. A real benchmark finishes and persists its result before release is requested.
2. GPU0 memory drops after a successful `/free` and the scheduler reports GPU0 as allocatable.
3. A subsequent real benchmark re-reserves GPU0 and reloads the required checkpoint.
4. No training or benchmark task overlaps the release transition.
5. Existing real LLM plan/review, worker, benchmark, evaluator, and hard-gate evidence remain intact.
