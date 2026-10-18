"""Real-CUDA profiling helpers for the existing Wan video evaluator."""

from __future__ import annotations

import gc
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


def _tensor_shape(value: Any) -> list[int] | None:
    if isinstance(value, Mapping):
        for item in value.values():
            found = _tensor_shape(item)
            if found is not None:
                return found
    elif isinstance(value, (tuple, list)):
        for item in value:
            found = _tensor_shape(item)
            if found is not None:
                return found
    elif hasattr(value, "shape"):
        try:
            return [int(item) for item in value.shape]
        except (TypeError, ValueError):
            return None
    return None


def _tensor_bytes(value: Any) -> int | None:
    if isinstance(value, Mapping):
        for item in value.values():
            found = _tensor_bytes(item)
            if found is not None:
                return found
    elif isinstance(value, (tuple, list)):
        for item in value:
            found = _tensor_bytes(item)
            if found is not None:
                return found
    elif hasattr(value, "numel") and hasattr(value, "element_size"):
        try:
            return int(value.numel()) * int(value.element_size())
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def _category(path: str, block_paths: Mapping[str, str]) -> str | None:
    if path == "":
        return "transformer_forward"
    if path in block_paths:
        return "transformer_block"
    for block_path, block_id in block_paths.items():
        prefix = block_path + "."
        if not path.startswith(prefix):
            continue
        local = path[len(prefix):].lower()
        if local == "attn1":
            return "self_attention"
        if local == "attn2":
            return "cross_attention"
        if local.startswith("attn1."):
            projection = local.rsplit(".", 1)[-1]
            if projection in {"to_q", "to_k", "to_v", "to_qkv"}:
                return "self_attention_%s_projection" % projection
            if local == "to_out.0" or local.endswith(".to_out.0"):
                return "self_attention_output_projection"
            return "self_attention"
        if local.startswith("attn2."):
            projection = local.rsplit(".", 1)[-1]
            if projection in {"to_q", "to_k", "to_v", "to_qkv"}:
                return "cross_attention_%s_projection" % projection
            if "to_out.0" in local:
                return "cross_attention_output_projection"
            return "cross_attention"
        if any(name in local.split(".") for name in ("ffn", "mlp", "feed_forward")):
            return "ffn"
        return None
    return None


def _parameter_count(module: Any, *, recurse: bool) -> int:
    try:
        return sum(int(item.numel()) for item in module.parameters(recurse=recurse))
    except (AttributeError, TypeError):
        return 0


def _event_record(torch: Any, device: Any) -> Any:
    event = torch.cuda.Event(enable_timing=True)
    event.record(torch.cuda.current_stream(device=device))
    return event


def _prepare_embeddings(evaluator: Any, pipeline: Any, torch: Any, prompt: str) -> Any:
    with torch.inference_mode():
        return evaluator._prepare_prompt_embeddings(
            pipeline,
            torch,
            prompt,
            device=evaluator.config.resolved_pipeline_device,
        )


class _ModuleTimer:
    def __init__(self, pipeline: Any, torch: Any, device: Any) -> None:
        self.pipeline = pipeline
        self.torch = torch
        self.device = device
        self.blocks: dict[str, str] = {}
        self.records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.active: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.step_index = 0
        self.handles: list[Any] = []
        transformer = pipeline.transformer
        from ..adapters.wan import build_wan_adapter

        adapter = build_wan_adapter("diffusers")
        # named_modules() below is relative to transformer, so use the same
        # paths here (DiffusersWanAdapter returns pipeline-prefixed paths when
        # passed the complete pipeline).
        for block in adapter.get_blocks(transformer):
            self.blocks[str(block.module_path)] = str(block.block_id)

        def root_pre(_module: Any, inputs: Any) -> None:
            self.active[""].append({
                "start": _event_record(self.torch, self.device),
                "step_index": self.step_index,
                "category": "transformer_forward",
                "input_shape": _tensor_shape(inputs),
                "input_bytes": _tensor_bytes(inputs),
            })

        def root_post(_module: Any, _inputs: Any, output: Any) -> None:
            stack = self.active.get("")
            if not stack:
                return
            record = stack.pop()
            record["end"] = _event_record(self.torch, self.device)
            record["output_shape"] = _tensor_shape(output)
            record["output_bytes"] = _tensor_bytes(output)
            self.records[""].append(record)

        self.handles.append(transformer.register_forward_pre_hook(root_pre))
        self.handles.append(transformer.register_forward_hook(root_post))
        for path, module in transformer.named_modules():
            if not path:
                continue  # root hooks above measure transformer.forward exactly once
            category = _category(str(path), self.blocks)
            if category is None:
                continue

            def pre(_module: Any, inputs: Any, _path: str = str(path), _category: str = category) -> None:
                self.active[_path].append({
                    "start": _event_record(self.torch, self.device),
                    "step_index": self.step_index,
                    "category": _category,
                    "input_shape": _tensor_shape(inputs),
                    "input_bytes": _tensor_bytes(inputs),
                })

            def post(_module: Any, _inputs: Any, output: Any, _path: str = str(path)) -> None:
                stack = self.active.get(_path)
                if not stack:
                    return
                record = stack.pop()
                record["end"] = _event_record(self.torch, self.device)
                record["output_shape"] = _tensor_shape(output)
                record["output_bytes"] = _tensor_bytes(output)
                self.records[_path].append(record)

            self.handles.append(module.register_forward_pre_hook(pre))
            self.handles.append(module.register_forward_hook(post))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def result(self) -> dict[str, Any]:
        self.torch.cuda.synchronize(self.device)
        modules = []
        categories: dict[str, list[float]] = defaultdict(list)
        by_step: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        module_by_step: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for path, values in self.records.items():
            durations = []
            for record in values:
                start, end = record.get("start"), record.get("end")
                if start is None or end is None:
                    continue
                try:
                    elapsed = float(start.elapsed_time(end))
                except (RuntimeError, ValueError):
                    continue
                durations.append(elapsed)
                category = str(record["category"])
                categories[category].append(elapsed)
                step_index = int(record["step_index"])
                by_step[step_index][category].append(elapsed)
                module_by_step[path][step_index].append(elapsed)
            if not durations:
                continue
            module = self.pipeline.transformer.get_submodule(path) if path else self.pipeline.transformer
            block_id = next((value for key, value in self.blocks.items() if path == key or path.startswith(key + ".")), None)
            first = values[0]
            modules.append({
                "module_path": path,
                "block_id": block_id,
                "category": first["category"],
                "calls": len(durations),
                "cuda_time_ms_total": sum(durations),
                "cuda_time_ms_median_per_call": float(statistics.median(durations)),
                "cuda_time_ms_by_step": {
                    str(step): sum(step_durations)
                    for step, step_durations in sorted(module_by_step[path].items())
                },
                "parameter_count": _parameter_count(module, recurse=(first["category"] in {"transformer_block", "self_attention", "cross_attention", "ffn"})),
                "input_shape": first.get("input_shape"),
                "output_shape": first.get("output_shape"),
                "input_activation_bytes": first.get("input_bytes"),
                "output_activation_bytes": first.get("output_bytes"),
                "timing_scope": "inclusive_forward_hook_cuda_event",
            })
        return {
            "module_timings": modules,
            "category_cuda_time_ms_total": {key: sum(values) for key, values in categories.items()},
            "category_cuda_time_ms_by_step": {
                str(step): {key: sum(values) for key, values in grouped.items()}
                for step, grouped in sorted(by_step.items())
            },
            "step_count_observed": len(by_step),
        }


def _profile_trace(
    evaluator: Any,
    pipeline: Any,
    torch: Any,
    prompt: str,
    seed: int,
    embeddings: Any,
    output_path: Path,
) -> dict[str, Any]:
    try:
        from torch.profiler import ProfilerActivity, profile, schedule
    except ImportError:
        return {"available": False, "reason": "torch_profiler_unavailable"}
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiler = None

    def on_trace_ready(value: Any) -> None:
        value.export_chrome_trace(str(output_path))

    try:
        with profile(
            activities=activities,
            schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            def callback(_pipe: Any, _step: int, _timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
                profiler.step()
                return callback_kwargs

            evaluator._generate(
                pipeline,
                torch,
                prompt,
                seed,
                prompt_embeds=embeddings[0],
                negative_prompt_embeds=embeddings[1],
                callback_on_step_end=callback,
            )
        events = profiler.key_averages()
        ops = []
        for item in events:
            cuda_us = float(getattr(item, "self_device_time_total", 0.0) or 0.0)
            if cuda_us <= 0.0:
                continue
            key = str(item.key)
            lower = key.lower()
            if "scaled_dot_product" in lower or "flash" in lower or "attention" in lower:
                category = "attention_kernel"
            elif any(token in lower for token in ("linear", "addmm", "mm", "gemm", "matmul")):
                category = "projection_or_matrix_multiply"
            elif any(token in lower for token in ("index", "gather", "scatter", "select", "copy", "cat", "contiguous")):
                category = "selection_or_data_movement"
            else:
                category = "other_cuda_operator"
            ops.append({
                "operator": key,
                "category": category,
                "calls": int(item.count),
                "self_cuda_time_us": cuda_us,
                "self_device_memory_bytes": int(getattr(item, "self_device_memory_usage", 0) or 0),
                "device_type": str(getattr(item, "device_type", "cuda")),
            })
        ops.sort(key=lambda item: float(item["self_cuda_time_us"]), reverse=True)
        return {
            "available": output_path.is_file(),
            "trace_path": str(output_path),
            "capture_scope": "second_denoising_step_after_one_profiler_warmup_step",
            "operator_summaries": ops[:40],
        }
    except (RuntimeError, OSError, TypeError, ValueError) as exc:
        return {"available": False, "reason": "%s: %s" % (type(exc).__name__, exc), "trace_path": str(output_path)}


def _module_diagnostic_pass(
    evaluator: Any,
    pipeline: Any,
    torch: Any,
    prompt: str,
    seed: int,
    embeddings: Any,
    device: Any,
) -> dict[str, Any]:
    timer = _ModuleTimer(pipeline, torch, device)
    step_times: list[dict[str, Any]] = []
    last = time.perf_counter()
    current_step = 0
    timer.step_index = 0

    def callback(_pipe: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        nonlocal last, current_step
        torch.cuda.synchronize(device)
        now = time.perf_counter()
        try:
            timestep_value = float(timestep.detach().float().cpu().item())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            timestep_value = None
        step_times.append({
            "step_index": int(step),
            "timestep": timestep_value,
            "wall_time_ms": (now - last) * 1000.0,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "allocated_bytes_at_callback": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes_at_callback": int(torch.cuda.memory_reserved(device)),
        })
        torch.cuda.reset_peak_memory_stats(device)
        last = now
        current_step = int(step) + 1
        timer.step_index = current_step
        return callback_kwargs

    original_decode = pipeline.vae.decode
    decode_events: list[tuple[Any, Any]] = []

    def timed_decode(*args: Any, **kwargs: Any) -> Any:
        start = _event_record(torch, device)
        result = original_decode(*args, **kwargs)
        decode_events.append((start, _event_record(torch, device)))
        return result

    pipeline.vae.decode = timed_decode
    original_postprocess = pipeline.video_processor.postprocess_video
    postprocess_times: list[float] = []

    def timed_postprocess(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = original_postprocess(*args, **kwargs)
        postprocess_times.append((time.perf_counter() - start) * 1000.0)
        return result

    pipeline.video_processor.postprocess_video = timed_postprocess
    try:
        evaluator._generate(
            pipeline,
            torch,
            prompt,
            seed,
            prompt_embeds=embeddings[0],
            negative_prompt_embeds=embeddings[1],
            callback_on_step_end=callback,
        )
        torch.cuda.synchronize(device)
    finally:
        timer.remove()
        pipeline.vae.decode = original_decode
        pipeline.video_processor.postprocess_video = original_postprocess
    decode_ms = []
    for start, end in decode_events:
        try:
            decode_ms.append(float(start.elapsed_time(end)))
        except (RuntimeError, ValueError):
            pass
    result = timer.result()
    step_peaks_allocated = [item["peak_allocated_bytes"] for item in step_times]
    step_peaks_reserved = [item["peak_reserved_bytes"] for item in step_times]
    result.update({
        "denoising_step_timings": step_times,
        "denoising_peak_allocated_bytes": max(step_peaks_allocated, default=None),
        "denoising_peak_reserved_bytes": max(step_peaks_reserved, default=None),
        "vae_decode_cuda_time_ms": sum(decode_ms) if decode_ms else None,
        "postprocess_wall_time_ms": sum(postprocess_times) if postprocess_times else None,
        "step_timing_scope": "synchronized_callback_intervals; first interval includes pipeline and scheduler setup",
        "step_memory_scope": "per_step allocator peaks reset at each synchronized callback; excludes text encoding",
    })
    return result


def profile_wan_execution(
    evaluator: Any,
    *,
    cases: Sequence[Mapping[str, Any]],
    prompt: str,
    seed: int,
    warmup: int,
    repeats: int,
    trace_dir: Path,
    head_gating_policy: Mapping[str, Any],
    on_case_complete: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Measure full Wan generation on CUDA, separating latency and trace passes."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("real Wan execution profiling requires CUDA")
    if not str(evaluator.config.device).startswith("cuda"):
        raise RuntimeError("real Wan execution profiling requires a CUDA generation device")
    if str(evaluator.config.device) != str(evaluator.config.resolved_pipeline_device):
        raise RuntimeError("Wan execution profiling currently requires one GPU for pipeline and generation")
    if warmup < 0 or repeats <= 0:
        raise ValueError("profile warmup must be non-negative and repeats positive")
    device = torch.device(evaluator.config.resolved_pipeline_device)
    trace_root = Path(trace_dir)
    variants = (
        {"name": "dense", "operator": None},
        {"name": "head_gating_0_75", "operator": dict(head_gating_policy)},
    )
    case_results: list[dict[str, Any]] = []
    for variant in variants:
        variant_cases = [
            case for case in cases
            if not case.get("variants") or variant["name"] in case.get("variants", [])
        ]
        if not variant_cases:
            continue
        pipeline, torch = evaluator._load_pipeline(evaluator.baseline.path)
        if variant["operator"] is not None:
            from ..adapters.wan import build_wan_adapter

            build_wan_adapter("diffusers", local_files_only=evaluator.config.local_files_only).apply_head_gating(
                pipeline,
                variant["operator"],
            )
        for case in variant_cases:
            shape = {
                "width": int(case["width"]),
                "height": int(case["height"]),
                "num_frames": int(case["num_frames"]),
                "num_inference_steps": int(case.get("num_inference_steps", evaluator._wan_spec["num_inference_steps"])),
            }
            case_warmup = int(case.get("warmup", warmup))
            case_repeats = int(case.get("repeats", repeats))
            if case_warmup < 0 or case_repeats <= 0:
                raise ValueError("each profile case requires non-negative warmup and positive repeats")
            original_spec = evaluator._wan_spec
            evaluator._wan_spec = {**dict(original_spec), **shape}
            try:
                text_samples: list[float] = []
                text_cuda_samples: list[float] = []
                generation_samples: list[float] = []
                end_to_end_samples: list[float] = []
                cuda_generation_samples: list[float] = []
                peak_allocated: list[int] = []
                peak_reserved: list[int] = []
                baseline_allocated = int(torch.cuda.memory_allocated(device))
                for index in range(case_warmup + case_repeats):
                    torch.cuda.reset_peak_memory_stats(device)
                    whole_start = time.perf_counter()
                    text_start = _event_record(torch, device)
                    text_wall_start = time.perf_counter()
                    embeddings = _prepare_embeddings(evaluator, pipeline, torch, prompt)
                    torch.cuda.synchronize(device)
                    text_wall = (time.perf_counter() - text_wall_start) * 1000.0
                    text_end = _event_record(torch, device)
                    if embeddings is None:
                        raise RuntimeError("Wan pipeline does not expose cached prompt embeddings")
                    cuda_start = _event_record(torch, device)
                    generation_start = time.perf_counter()
                    frames = evaluator._generate_case(pipeline, torch, prompt, seed, embeddings)
                    torch.cuda.synchronize(device)
                    generation_wall = (time.perf_counter() - generation_start) * 1000.0
                    cuda_end = _event_record(torch, device)
                    end_to_end = (time.perf_counter() - whole_start) * 1000.0
                    if index >= case_warmup:
                        text_samples.append(text_wall)
                        text_cuda_samples.append(float(text_start.elapsed_time(text_end)))
                        generation_samples.append(generation_wall)
                        end_to_end_samples.append(end_to_end)
                        cuda_generation_samples.append(float(cuda_start.elapsed_time(cuda_end)))
                        peak_allocated.append(int(torch.cuda.max_memory_allocated(device)))
                        peak_reserved.append(int(torch.cuda.max_memory_reserved(device)))
                    del frames, embeddings, text_start, text_end
                torch.cuda.synchronize(device)

                # One instrumented pass attributes inclusive module and per-step time.
                torch.cuda.reset_peak_memory_stats(device)
                embeddings = _prepare_embeddings(evaluator, pipeline, torch, prompt)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                diagnostic = _module_diagnostic_pass(evaluator, pipeline, torch, prompt, seed, embeddings, device)
                del embeddings
                pass_peak_allocated = int(torch.cuda.max_memory_allocated(device))
                pass_peak_reserved = int(torch.cuda.max_memory_reserved(device))
                diagnostic["peak_allocated_bytes"] = max(
                    pass_peak_allocated,
                    int(diagnostic.get("denoising_peak_allocated_bytes") or 0),
                )
                diagnostic["peak_reserved_bytes"] = max(
                    pass_peak_reserved,
                    int(diagnostic.get("denoising_peak_reserved_bytes") or 0),
                )
                diagnostic["activation_peak_above_model_bytes"] = max(0, diagnostic["peak_allocated_bytes"] - baseline_allocated)

                trace = {"available": False, "reason": "not_requested"}
                if bool(case.get("capture_kernel_trace", False)):
                    embeddings = _prepare_embeddings(evaluator, pipeline, torch, prompt)
                    trace_name = "%s-%dx%d-%df.json" % (
                        variant["name"], shape["width"], shape["height"], shape["num_frames"]
                    )
                    trace_path = trace_root / trace_name
                    trace = _profile_trace(evaluator, pipeline, torch, prompt, seed, embeddings, trace_path)
                    del embeddings

                transformer = pipeline.transformer
                head_stats = dict(getattr(transformer, "_harness4h3_head_gating_stats", {}) or {})
                case_results.append({
                    "variant": variant["name"],
                    "operator": variant["operator"],
                    "recipe": shape,
                    "prompt": prompt,
                    "seed": int(seed),
                    "warmup": case_warmup,
                    "repeats": case_repeats,
                    "latency_samples": {
                        "text_encoder_wall_ms": text_samples,
                        "text_encoder_cuda_event_ms": text_cuda_samples,
                        "denoising_vae_postprocess_wall_ms": generation_samples,
                        "denoising_vae_cuda_event_ms": cuda_generation_samples,
                        "full_generation_with_text_wall_ms": end_to_end_samples,
                    },
                    "latency_median_ms": {
                        "text_encoder_wall_ms": float(statistics.median(text_samples)),
                        "text_encoder_cuda_event_ms": float(statistics.median(text_cuda_samples)),
                        "denoising_vae_postprocess_wall_ms": float(statistics.median(generation_samples)),
                        "denoising_vae_cuda_event_ms": float(statistics.median(cuda_generation_samples)),
                        "full_generation_with_text_wall_ms": float(statistics.median(end_to_end_samples)),
                    },
                    "memory": {
                        "peak_allocated_bytes_median": int(statistics.median(peak_allocated)),
                        "peak_reserved_bytes_median": int(statistics.median(peak_reserved)),
                        "model_resident_allocated_bytes": baseline_allocated,
                        "measurement_scope": str(device),
                    },
                    "parameter_count": int(sum(int(parameter.numel()) for parameter in transformer.parameters())),
                    "head_gating_execution": head_stats,
                    "diagnostic_pass": diagnostic,
                    "kernel_trace": trace,
                })
                if on_case_complete is not None:
                    on_case_complete(case_results[-1])
            finally:
                evaluator._wan_spec = original_spec
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    expected_case_count = sum(
        sum(1 for variant in variants if not case.get("variants") or variant["name"] in case.get("variants", []))
        for case in cases
    )
    return {
        "execution_schema_version": 1,
        "available": True,
        "offline_simulation": False,
        "partial": len(case_results) != expected_case_count,
        "expected_case_count": expected_case_count,
        "observed_case_count": len(case_results),
        "measurement_backend": "torch.cuda.Event+torch.profiler",
        "device": str(device),
        "pipeline_device": str(evaluator.config.resolved_pipeline_device),
        "dtype": str(evaluator.config.dtype),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "diffusers_version": __import__("diffusers").__version__,
        "checkpoint_sha256": evaluator.baseline.sha256,
        "source_revision": evaluator.config.source_revision,
        "latency_method": "cuda_events_and_wall_clock_after_synchronization",
        "diagnostic_method": "inclusive_module_forward_hooks_and_one_step_torch_profiler_trace",
        "cases": case_results,
    }
