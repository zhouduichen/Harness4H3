# 4×L40 Training Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add safe, testable four-L40 device, distributed smoke-recipe, and worker configuration prerequisites without enabling or implementing real H3 training.

**Architecture:** Extend the existing read-only device preflight with fixed `nvidia-smi`, Linux RAM, and PyTorch/CUDA observations plus operation-scoped path checks. Keep expected hardware requirements separate from the A1-T0 experimental recipe and trusted worker argv. A pending/unconfigured host always remains blocked, and enabling `recovery_finetune` requires a later reviewed trainer smoke result.

**Tech Stack:** Python 3.9+, PyYAML, argparse, subprocess, pathlib, pytest, YAML, JSON.

## Global Constraints

- Do not modify Harness4H3-v1.0 Controller, schema, evaluator, archive, trajectory, acceptance, or TargetProfile contracts.
- Do not implement an H3 trainer or start A1-T0.
- Do not execute any command supplied by a device profile.
- Use only the fixed `nvidia-smi` query for GPU discovery.
- Treat four L40 GPUs and at least 40 GiB per GPU as requirements, not measured facts.
- Treat 128 GiB system RAM as a minimum deployment requirement, not a measured fact.
- Keep `recovery_finetune`, `prune`, and `distill` disabled.
- Do not use fixture/mock trainers in any runtime configuration.
- Preserve the user's uncommitted `research/experiments/m6_campaign.py` and `research/experiments/m6_runtime_recipe.py` changes.

---

### Task 1: Extend read-only hardware and operation-path preflight

**Files:**
- Modify: `tools/device_preflight.py`
- Modify: `tests/unit/test_device_preflight.py`

**Interfaces:**
- Consumes: optional `hardware.gpu_name_contains`, `hardware.gpu_count`, `hardware.min_vram_gib_per_gpu`, `hardware.min_system_ram_gib`, `hardware.distributed_backend`, and path-entry `required_for` lists.
- Produces: `preflight(profile_path: Path, operator: Optional[str], check_services: bool = True, timeout_s: float = 3.0, check_hardware: bool = True) -> Dict[str, Any]` and stable `hardware.*`/`runtime.torch` checks.

- [x] **Step 1: Add failing test helpers and passing four-GPU test**

Add these helpers to `tests/unit/test_device_preflight.py`:

```python
def distributed_profile(tmp_path: Path) -> dict:
    raw = valid_profile(tmp_path)
    raw["hardware"] = {
        "gpu_name_contains": "NVIDIA L40",
        "gpu_count": 4,
        "min_vram_gib_per_gpu": 40,
        "min_system_ram_gib": 128,
        "distributed_backend": "nccl",
    }
    raw["capabilities"]["recovery_finetune"] = {"enabled": True, "reason": "test only"}
    return raw


def install_hardware_observations(tmp_path, monkeypatch, gpu_lines: str, ram_gib: int = 256):
    class Completed:
        returncode = 0
        stdout = gpu_lines
        stderr = ""

    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: %d kB\n" % (ram_gib * 1024 * 1024), encoding="utf-8")
    monkeypatch.setattr(device_preflight, "run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr(device_preflight, "MEMINFO_PATH", meminfo)
    monkeypatch.setattr(
        device_preflight,
        "_torch_observation",
        lambda: {"version": "2.11.0", "cuda": "12.8", "available": True, "device_count": 4},
    )
```

Add the test:

```python
def test_preflight_accepts_four_matching_l40_gpus(tmp_path, monkeypatch):
    install_hardware_observations(
        tmp_path,
        monkeypatch,
        "NVIDIA L40, 46068\nNVIDIA L40, 46068\nNVIDIA L40, 46068\nNVIDIA L40, 46068\n",
    )
    result = device_preflight.preflight(
        write_profile(tmp_path, distributed_profile(tmp_path)),
        "recovery_finetune",
        check_services=False,
    )
    assert result["status"] == "ready"
    assert check(result, "hardware.gpu_count")["status"] == "passed"
    assert check(result, "hardware.gpu_name")["status"] == "passed"
    assert check(result, "hardware.gpu_vram")["status"] == "passed"
    assert check(result, "hardware.system_ram")["status"] == "passed"
    assert check(result, "runtime.torch")["status"] == "passed"
```

- [x] **Step 2: Run the focused test and verify failure**

Run:

```bash
python3 -m pytest -q tests/unit/test_device_preflight.py::test_preflight_accepts_four_matching_l40_gpus
```

Expected: fail because `run`, `MEMINFO_PATH`, `_torch_observation`, and hardware checks do not exist.

- [x] **Step 3: Implement fixed hardware observations**

In `tools/device_preflight.py`, import `subprocess.run`, `subprocess.PIPE`, and `sys`; define:

```python
MEMINFO_PATH = Path("/proc/meminfo")
NVIDIA_SMI_ARGV = (
    "nvidia-smi",
    "--query-gpu=name,memory.total",
    "--format=csv,noheader,nounits",
)


def _gpu_observations() -> List[Dict[str, Any]]:
    completed = run(NVIDIA_SMI_ARGV, stdin=PIPE, capture_output=True, text=True, check=False, timeout=10)
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi exited with status %d: %s" % (completed.returncode, completed.stderr.strip()))
    observations = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        name, memory_mib = line.rsplit(",", 1)
        observations.append({"name": name.strip(), "memory_gib": float(memory_mib.strip()) / 1024.0})
    if not observations:
        raise ValueError("nvidia-smi returned no GPUs")
    return observations


def _system_ram_gib() -> float:
    for line in MEMINFO_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    raise ValueError("MemTotal is missing from /proc/meminfo")


def _torch_observation() -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc
    return {
        "version": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
```

Catch observation failures and emit stable failed checks rather than raising
past `preflight`. Compare all GPUs against the declared count, name substring,
and minimum GiB. Require PyTorch CUDA availability and the same visible device
count as `hardware.gpu_count`.

- [x] **Step 4: Implement hardware skipping and operation-scoped paths**

Add `check_hardware` to `preflight`. When hardware requirements exist and it
is false, append `skipped` entries for all five checks and force overall status
to `blocked`. Change path selection to:

```python
required_for = entry.get("required_for", [])
if operator and required_for and operator not in required_for:
    continue
```

Add CLI flag:

```python
parser.add_argument("--skip-hardware", action="store_true")
```

and pass `check_hardware=not args.skip_hardware`.

- [x] **Step 5: Add negative hardware/path tests**

Add parameterized cases for three hardware mismatches:

```python
@pytest.mark.parametrize(
    ("gpu_lines", "failed_check"),
    [
        ("NVIDIA L40, 46068\n" * 3, "hardware.gpu_count"),
        (("NVIDIA A100, 46068\n" + "NVIDIA L40, 46068\n" * 3), "hardware.gpu_name"),
        (("NVIDIA L40, 39000\n" + "NVIDIA L40, 46068\n" * 3), "hardware.gpu_vram"),
    ],
)
def test_preflight_rejects_gpu_mismatch(tmp_path, monkeypatch, gpu_lines, failed_check):
    install_hardware_observations(tmp_path, monkeypatch, gpu_lines)
    result = device_preflight.preflight(
        write_profile(tmp_path, distributed_profile(tmp_path)),
        "recovery_finetune",
        check_services=False,
    )
    assert result["status"] == "blocked"
    assert check(result, failed_check)["status"] == "failed"
```

Also add explicit tests for missing `nvidia-smi`, malformed output,
64-GiB RAM, unavailable PyTorch CUDA, `--skip-hardware`, and two required paths
with disjoint `required_for: [recovery_finetune]` and
`required_for: [benchmark]` lists.

- [x] **Step 6: Run focused tests and commit**

Run:

```bash
python3 -m pytest -q tests/unit/test_device_preflight.py
python3 -m compileall -q tools/device_preflight.py tests/unit/test_device_preflight.py
git diff --check
git add tools/device_preflight.py tests/unit/test_device_preflight.py
git commit -m "feat: probe distributed GPU training prerequisites"
```

Expected: all device-preflight tests pass; no M6 file is staged.

---

### Task 2: Add pending 4×L40 configuration contracts

**Files:**
- Create: `configs/devices/l40x4-server.yaml`
- Create: `configs/experiments/a1-t0-l40x4.yaml`
- Create: `configs/a1-worker.l40x4.example.json`
- Create: `tests/unit/test_l40x4_configs.py`

**Interfaces:**
- Consumes: the extended device-profile preflight and existing `h3_model_worker.py` fixed-argv contract.
- Produces: one pending device requirement profile, one immutable one-step FSDP recipe, and one non-fixture Linux worker example.

- [x] **Step 1: Write failing configuration-contract tests**

Create `tests/unit/test_l40x4_configs.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def test_l40x4_device_profile_is_pending_and_safe():
    raw = load_yaml("configs/devices/l40x4-server.yaml")
    assert raw["verification"]["status"] == "pending"
    assert raw["hardware"]["gpu_count"] == 4
    assert raw["hardware"]["gpu_name_contains"] == "NVIDIA L40"
    assert raw["hardware"]["min_vram_gib_per_gpu"] == 40
    assert raw["hardware"]["min_system_ram_gib"] == 128
    assert raw["hardware"]["distributed_backend"] == "nccl"
    assert raw["capabilities"]["recovery_finetune"]["enabled"] is False
    assert raw["capabilities"]["prune"]["enabled"] is False
    assert raw["capabilities"]["distill"]["enabled"] is False


def test_l40x4_smoke_recipe_is_bounded_and_sharded():
    raw = load_yaml("configs/experiments/a1-t0-l40x4.yaml")
    assert raw["operator"] == "recovery_finetune"
    assert raw["distributed"] == {
        "launcher": "torchrun",
        "strategy": "fsdp_full_shard",
        "world_size": 4,
        "backend": "nccl",
        "use_orig_params": True,
    }
    assert raw["model"]["dtype"] == "bfloat16"
    assert raw["model"]["trainable_scope"] == "heads"
    assert raw["memory"]["micro_batch_size"] == 1
    assert raw["optimization"]["max_steps"] == 1
    assert raw["data"]["sample_count"] == 1
    assert raw["checkpoint"]["require_diffusers_reload"] is True
    assert raw["checkpoint"]["require_comfyui_reload"] is True


def test_l40x4_worker_uses_fixed_four_rank_linux_argv():
    raw = json.loads((ROOT / "configs/a1-worker.l40x4.example.json").read_text(encoding="utf-8"))
    command = raw["trainer_command"]
    assert command[0] == "/opt/h3-training/.venv/bin/torchrun"
    assert "--nproc_per_node=4" in command
    assert "/opt/h3-training/train_worker.py" in command
    assert "/opt/Harness4H3/configs/experiments/a1-t0-l40x4.yaml" in command
    assert not any("fixture" in item or "mock" in item for item in command)
    assert raw["deploy_model_dir"] == "/opt/ComfyUI/models/diffusion_models"
```

- [x] **Step 2: Run tests and verify missing-file failures**

Run:

```bash
python3 -m pytest -q tests/unit/test_l40x4_configs.py
```

Expected: three failures because the configuration files do not exist.

- [x] **Step 3: Create the pending device profile**

Copy the exact device-profile fields from the approved design. Add required
paths for `/opt/Harness4H3`, `/models/MiniMax-H3`,
`/opt/h3-training/train_worker.py`, the checked-in smoke recipe,
`/opt/ComfyUI`, and the ComfyUI deployment directory. Use `required_for` to
separate `recovery_finetune` and `benchmark` paths. Configure localhost
ComfyUI and Controller health endpoints. Keep all real capabilities disabled
with concrete reasons.

- [x] **Step 4: Create the fixed smoke recipe and worker example**

Create the complete recipe shown in the approved design, including finite
loss, non-zero gradient, optimizer-step, parent immutability, changed expected
tensors, frozen-tensor integrity, per-rank peak-memory, child hash, and reload
evidence requirements under an `evidence.required` list.

Create the worker JSON with:

```json
{
  "trainer_command": [
    "/opt/h3-training/.venv/bin/torchrun",
    "--standalone",
    "--nproc_per_node=4",
    "/opt/h3-training/train_worker.py",
    "--config",
    "/opt/Harness4H3/configs/experiments/a1-t0-l40x4.yaml"
  ],
  "trainer_timeout_s": 7200,
  "deploy_model_dir": "/opt/ComfyUI/models/diffusion_models"
}
```

- [x] **Step 5: Prove the unconfigured profile blocks safely**

Run:

```bash
python3 tools/device_preflight.py --profile configs/devices/l40x4-server.yaml --operator recovery_finetune --skip-services --json
```

Expected: exit `2`; result includes disabled capability and missing/unmatched
local prerequisites. It must not contain `status: ready`.

- [x] **Step 6: Run configuration tests and commit**

Run:

```bash
python3 -m pytest -q tests/unit/test_l40x4_configs.py tests/unit/test_device_preflight.py
git diff --check
git add configs/devices/l40x4-server.yaml configs/experiments/a1-t0-l40x4.yaml configs/a1-worker.l40x4.example.json tests/unit/test_l40x4_configs.py
git commit -m "config: prepare four-L40 A1 smoke prerequisites"
```

Expected: all focused tests pass; only configuration and its tests are
committed.

---

### Task 3: Document setup, verify, and archive the completed design

**Files:**
- Modify: `docs/device-porting.md`
- Modify: `research/README.md`
- Move: `docs/superpowers/specs/2026-09-13-l40x4-training-preflight-design.md` to `research/history/specs/2026-09-13-l40x4-training-preflight-design.md`
- Move: `docs/superpowers/plans/2026-09-13-l40x4-training-preflight.md` to `research/history/plans/2026-09-13-l40x4-training-preflight.md`

**Interfaces:**
- Consumes: the pending profile, smoke recipe, worker example, and preflight command.
- Produces: a reproducible host-setup checklist and honest research-status record.

- [x] **Step 1: Document the 4×L40 setup gate**

Add a `4×L40 pending training host` section to `docs/device-porting.md` that
links all three configurations and gives this order:

```text
install host and NVIDIA driver
→ run nvidia-smi and record topology
→ install Python/PyTorch CUDA environment
→ place official Diffusers H3 assets
→ deploy trainer without enabling it
→ run device preflight and persist JSON
→ review measured evidence
→ implement/run isolated A1-T0
→ enable recovery_finetune only after A1-T0 passes
```

State that ordinary DDP is invalid because it replicates the full transformer.

- [x] **Step 2: Update research capability status**

In `research/README.md`, add that a four-L40 FSDP preflight configuration is
prepared but unverified; no forward/backward, optimizer step, or child exists
from that host.

- [x] **Step 3: Run full verification**

Run:

```bash
python3 -m pytest -q
python3 -m compileall -q harness4h3 research tools tests
python3 tools/device_preflight.py --profile configs/devices/l40x4-server.yaml --operator recovery_finetune --skip-services --json
git diff --check
```

Expected: full tests and compileall pass; preflight returns exit `2` and a
machine-readable blocked result.

- [x] **Step 4: Mark plan/spec complete and archive them**

Change the current design status to `implemented and archived`, mark every
checkbox in this plan complete, create `research/history/plans` and
`research/history/specs` if needed, and move the two current files with
`git mv`.

- [x] **Step 5: Commit documentation/history and inspect the worktree**

Run:

```bash
git add docs/device-porting.md research/README.md research/history docs/superpowers
git commit -m "docs: record four-L40 training preflight"
git status --short
```

Expected: only the user's pre-existing M6 working-tree changes remain.
