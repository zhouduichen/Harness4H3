# Harness4H3 Project Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Reorganize Harness4H3 into a portable research harness whose reusable system, reproducible experiments, device facts, evidence, and historical material are immediately distinguishable.

**Architecture:** Keep the frozen `harness4h3/` implementation intact except for import paths required by the research-package move. Move phase-specific programs into an installable `research.experiments` package, add a declarative device profile with a small standard-library preflight tool, and make the README plus active guides describe the research question, protocol, capability boundaries, and reproduction path. Preserve measured evidence and the user's current M6 edits.

**Tech Stack:** Python 3.9+, PyYAML, argparse, urllib, pytest, Markdown, Git.

## Global Constraints

- `Harness4H3-v1.0` controller protocol, schemas, evaluator, archive, trajectory, acceptance policy, and TargetProfile semantics remain unchanged.
- Do not add an orchestration abstraction, plugin framework, trainer, pruning backend, or distillation backend.
- Do not claim that real model-changing H3 training is available.
- Keep `harness4h3` as the only installed console script.
- Preserve all real evidence and failure/rejection records.
- Preserve the existing working-tree changes in `experiments/m6_campaign.py` and `experiments/m6_runtime_recipe.py` byte-for-byte apart from their paths and required import/evidence-path updates.
- Add no dependency solely for repository organization.
- Use research language only for explicit hypotheses, protocols, observations, and bounded claims.

---

### Task 1: Establish the research package without losing user work

**Files:**
- Create: `research/__init__.py`
- Move: `experiments/__init__.py` to `research/experiments/__init__.py`
- Move: `experiments/a0_model_evolution.py` to `research/experiments/a0_model_evolution.py`
- Move: `experiments/a1_real_evolution.py` to `research/experiments/a1_real_evolution.py`
- Move: `experiments/m5_validation.py` to `research/experiments/m5_validation.py`
- Move: `experiments/m6_campaign.py` to `research/experiments/m6_campaign.py`
- Move: `experiments/m6_runtime_memory.py` to `research/experiments/m6_runtime_memory.py`
- Move: `experiments/m6_runtime_recipe.py` to `research/experiments/m6_runtime_recipe.py`
- Move: `experiments/power_study.py` to `research/experiments/power_study.py`
- Modify: `harness4h3/cli.py`
- Modify: `pyproject.toml`
- Modify: tests importing `experiments.*`

**Interfaces:**
- Consumes: Existing experiment functions and CLI command behavior.
- Produces: Importable modules under `research.experiments.*`; installed distributions contain `harness4h3*` and `research*`.

- [x] **Step 1: Record the dirty-file content hashes**

Run:

```bash
shasum -a 256 experiments/m6_campaign.py experiments/m6_runtime_recipe.py
git diff --binary -- experiments/m6_campaign.py experiments/m6_runtime_recipe.py > /tmp/harness4h3-m6-before.patch
```

Expected: two SHA-256 values and a backup patch outside the repository.

- [x] **Step 2: Move the experiment package with Git history**

Run:

```bash
mkdir -p research/experiments
git mv experiments/__init__.py research/experiments/__init__.py
git mv experiments/a0_model_evolution.py research/experiments/a0_model_evolution.py
git mv experiments/a1_real_evolution.py research/experiments/a1_real_evolution.py
git mv experiments/m5_validation.py research/experiments/m5_validation.py
git mv experiments/m6_campaign.py research/experiments/m6_campaign.py
git mv experiments/m6_runtime_memory.py research/experiments/m6_runtime_memory.py
git mv experiments/m6_runtime_recipe.py research/experiments/m6_runtime_recipe.py
git mv experiments/power_study.py research/experiments/power_study.py
```

Create `research/__init__.py` containing:

```python
"""Reproducible Harness4H3 research programs and records."""
```

- [x] **Step 3: Update executable imports and package discovery**

Mechanically replace the import prefix `experiments.` with
`research.experiments.` in `harness4h3/cli.py`, all moved research programs,
and these tests:

```text
tests/test_power_study.py
tests/unit/test_a0_model_evolution.py
tests/unit/test_a1_real_evolution.py
tests/unit/test_m6_campaign.py
tests/unit/test_m6_runtime_recipe.py
```

Set package discovery in `pyproject.toml` to:

```toml
[tool.setuptools.packages.find]
include = ["harness4h3*", "research*"]
```

- [x] **Step 4: Prove imports and CLI parsing still work**

Run:

```bash
.venv/bin/python -m pytest -q tests/unit/test_a0_model_evolution.py tests/unit/test_a1_real_evolution.py tests/unit/test_m6_campaign.py tests/unit/test_m6_runtime_recipe.py tests/test_power_study.py tests/test_cli.py
.venv/bin/python -m harness4h3 --help
```

Expected: selected tests pass; help lists all existing commands.

- [x] **Step 5: Verify user changes survived the move and commit**

Run:

```bash
git diff --binary -- research/experiments/m6_campaign.py research/experiments/m6_runtime_recipe.py
git diff --check
git add research harness4h3/cli.py pyproject.toml tests
git commit -m "refactor: separate research experiments from harness core"
```

Expected: the original M6 additions remain present; the commit records renames rather than delete/recreate churn.

---

### Task 2: Add a declarative device profile and read-only preflight

**Files:**
- Create: `configs/devices/rtx5080-laptop.yaml`
- Create: `tools/device_preflight.py`
- Create: `tests/unit/test_device_preflight.py`

**Interfaces:**
- Consumes: A YAML mapping with `schema_version`, `id`, `platform`, `hardware`, `paths`, `services`, `limits`, `formats`, and `capabilities`.
- Produces: `preflight(profile_path: Path, operator: str | None, check_services: bool = True) -> dict`; CLI JSON with `status`, `profile_id`, `operator`, and stable `checks` entries.

- [x] **Step 1: Write failing validation and capability tests**

Create the test module with this concrete structure (the implementation may
return additional checks, but these names and outcomes are fixed):

```python
from pathlib import Path
from urllib.error import URLError

import yaml

import tools.device_preflight as device_preflight


class Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def write_profile(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "device.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def valid_profile(tmp_path: Path) -> dict:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return {
        "schema_version": 1,
        "id": "test-device",
        "platform": {"os": "test", "python": "3.12"},
        "hardware": {"gpu_name": "test-gpu", "gpu_count": 1, "vram_gb": 16, "ram_gb": 32},
        "paths": {"workspace": {"path": str(workspace), "required": True}},
        "services": {
            "comfyui": {
                "url": "http://127.0.0.1:8188",
                "health_path": "/system_stats",
                "required_for": ["benchmark"],
            }
        },
        "limits": {"max_peak_vram_gb": 16},
        "formats": ["safetensors"],
        "capabilities": {
            "benchmark": {"enabled": True, "reason": "test service"},
            "recovery_finetune": {"enabled": False, "reason": "no trainer"},
        },
    }


def check(result: dict, name: str) -> dict:
    return next(item for item in result["checks"] if item["name"] == name)


def test_preflight_accepts_complete_profile_with_existing_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(device_preflight, "urlopen", lambda request, timeout: Response())
    result = device_preflight.preflight(write_profile(tmp_path, valid_profile(tmp_path)), "benchmark")
    assert result["status"] == "ready"
    assert check(result, "profile.schema")["status"] == "passed"
    assert check(result, "capability.benchmark")["status"] == "passed"
    assert check(result, "path.workspace")["status"] == "passed"
    assert check(result, "service.comfyui")["status"] == "passed"


def test_preflight_reports_missing_required_field(tmp_path):
    raw = valid_profile(tmp_path)
    del raw["hardware"]
    result = device_preflight.preflight(write_profile(tmp_path, raw), None, check_services=False)
    assert result["status"] == "blocked"
    assert check(result, "profile.schema")["status"] == "failed"


def test_preflight_reports_disabled_operator_reason(tmp_path):
    result = device_preflight.preflight(
        write_profile(tmp_path, valid_profile(tmp_path)), "recovery_finetune", check_services=False
    )
    assert result["status"] == "blocked"
    assert check(result, "capability.recovery_finetune")["detail"] == "no trainer"


def test_preflight_reports_missing_path(tmp_path):
    raw = valid_profile(tmp_path)
    raw["paths"]["workspace"]["path"] = str(tmp_path / "missing")
    result = device_preflight.preflight(write_profile(tmp_path, raw), "benchmark", check_services=False)
    assert result["status"] == "blocked"
    assert check(result, "path.workspace")["status"] == "failed"


def test_preflight_reports_unreachable_required_service(tmp_path, monkeypatch):
    def unavailable(request, timeout):
        raise URLError("offline")

    monkeypatch.setattr(device_preflight, "urlopen", unavailable)
    result = device_preflight.preflight(write_profile(tmp_path, valid_profile(tmp_path)), "benchmark")
    assert result["status"] == "blocked"
    assert check(result, "service.comfyui")["status"] == "failed"
```

Assertions use stable check names: `profile.schema`, `capability.<operator>`, `path.<name>`, and `service.<name>`; success returns `status == "ready"`, any failed check returns `status == "blocked"`.

- [x] **Step 2: Run the tests to verify the tool is absent**

Run:

```bash
.venv/bin/python -m pytest -q tests/unit/test_device_preflight.py
```

Expected: collection fails because `tools/device_preflight.py` does not exist.

- [x] **Step 3: Implement the minimal preflight tool**

Use `yaml.safe_load`, `pathlib.Path`, `urllib.parse.urljoin`, and `urllib.request.urlopen`. Validate the exact required top-level mappings, resolve relative paths against the repository root, check only paths marked `required: true`, check only services whose `required_for` contains the requested operator, and never execute a model or experiment. Return checks in this shape:

```python
{
    "name": "capability.recovery_finetune",
    "status": "failed",
    "detail": "official BF16 transformer exceeds verified VRAM and RAM",
}
```

CLI:

```text
python tools/device_preflight.py --profile PATH [--operator NAME] [--skip-services] [--json]
```

Exit `0` for `ready`, `2` for `blocked` or invalid input.

- [x] **Step 4: Add the measured RTX 5080 profile**

The profile records Windows, one RTX 5080 Laptop GPU, `15.92` GiB VRAM, approximately `31.45` GiB RAM, the verified checkpoint manifest, ComfyUI/controller health endpoints, accepted `safetensors`/`gguf` formats, and explicit capability declarations:

```yaml
capabilities:
  inference:
    enabled: true
    reason: verified through the existing H3 ComfyUI backend
  benchmark:
    enabled: true
    reason: measured benchmark evidence is archived
  recovery_finetune:
    enabled: false
    reason: official BF16 transformer exceeds verified VRAM and RAM
  prune:
    enabled: false
    reason: no real H3 pruning backend is registered
  distill:
    enabled: false
    reason: no real H3 teacher/student training backend is registered
```

- [x] **Step 5: Run focused tests and commit**

Run:

```bash
.venv/bin/python -m pytest -q tests/unit/test_device_preflight.py
.venv/bin/python tools/device_preflight.py --profile configs/devices/rtx5080-laptop.yaml --operator recovery_finetune --skip-services --json
git add configs/devices tools/device_preflight.py tests/unit/test_device_preflight.py
git commit -m "feat: add device capability preflight"
```

Expected: tests pass; the real profile command exits `2` and reports the declared training blocker rather than starting work.

---

### Task 3: Separate evidence and historical research material

**Files:**
- Move: `docs/real-experiments/` to `research/evidence/real-experiments/`
- Move: `docs/experience/` to `research/evidence/design-genes/`
- Move: `docs/evogen-*.md` to `research/history/`
- Move: `REUSE_MATRIX.md` to `research/history/reuse-matrix.md`
- Modify: runtime research programs that load Design Genes
- Modify: active documentation references outside immutable evidence files
- Modify: `.gitignore`
- Remove: untracked `docs/.DS_Store`

**Interfaces:**
- Consumes: Existing evidence bytes and research script lookups.
- Produces: One evidence tree and one history tree; active code resolves Design Genes at their new paths.

- [x] **Step 1: Record evidence hashes and move files with Git**

Run:

```bash
git ls-files docs/real-experiments docs/experience | sort | xargs shasum -a 256 > /tmp/harness4h3-evidence-before.sha256
mkdir -p research/evidence research/history
git mv docs/real-experiments research/evidence/real-experiments
git mv docs/experience research/evidence/design-genes
git mv docs/evogen-a0-model-evolution.md research/history/
git mv docs/evogen-a1-real-model-evolution.md research/history/
git mv docs/evogen-rsi-research.md research/history/
git mv REUSE_MATRIX.md research/history/reuse-matrix.md
```

- [x] **Step 2: Update active code to the new Design Gene locations**

Use repository-root-relative paths:

```python
root / "research/evidence/design-genes/design-gene-h3-nvfp4.json"
root / "research/evidence/design-genes/design-gene-m6-vae-tiling.json"
```

Do not edit the moved evidence records themselves.

- [x] **Step 3: Remove local metadata and prevent recurrence**

Delete only `/Users/huangjiahao/MinMax-H3/Harness4H3/docs/.DS_Store` and add this line to `.gitignore`:

```gitignore
.DS_Store
```

- [x] **Step 4: Verify evidence bytes and research imports**

Create an after-hash list using the moved paths and compare hash columns, not path columns. Run:

```bash
.venv/bin/python -m pytest -q tests/unit/test_m6_runtime_recipe.py tests/unit/test_m6_campaign.py
git diff --check
```

Expected: evidence hashes are unchanged and focused tests pass.

- [x] **Step 5: Commit the evidence/history separation**

Run:

```bash
git add .gitignore research/evidence research/history research/experiments
git commit -m "docs: separate research evidence and history"
```

---

### Task 4: Present the repository as a reproducible research system

**Files:**
- Rewrite: `README.md`
- Rewrite: `docs/architecture.md`
- Create: `docs/quickstart.md`
- Create: `docs/device-porting.md`
- Create: `docs/optimization-flow.md`
- Create: `docs/operator-contract.md`
- Create: `research/README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: Existing CLI, device preflight, benchmark command, worker contract, experiment/evidence locations.
- Produces: A short root onboarding path and a research index organized as hypothesis → protocol → implementation → evidence → limitations.

- [x] **Step 1: Rewrite the root positioning and capability table**

The README must state:

```text
Research question: can a bounded controller select device-aware H3 model/runtime interventions and retain only candidates supported by independent measurements?
Status: real inference and benchmark verified; runtime optimization measured; real model-changing training blocked; pruning/distillation are contracts, not implemented backends.
```

It then links to quickstart, device porting, optimization flow, architecture, operator contract, research index, evidence, and history. Keep installation plus one offline command, one preflight command, and one real benchmark command. Remove phase-by-phase result narration from the root.

- [x] **Step 2: Write the active guides around reproducibility**

Each guide has one responsibility:

- `docs/quickstart.md`: install, `validate-config`, fake closed-loop smoke, checkpoint inspect, preflight, and benchmark.
- `docs/device-porting.md`: copy repository, create environment, copy/edit the device profile, start external services, run preflight, and interpret `ready`/`blocked`.
- `docs/optimization-flow.md`: baseline → plan → validation → operator → candidate → benchmark → independent evaluation → archive/trajectory → next plan; separate model-level from runtime-level interventions.
- `docs/operator-contract.md`: structured request/result, parent immutability, child authenticity, logs/metrics/costs, and stable failures; explicitly label trainer/prune/distill implementations as missing.
- `docs/architecture.md`: frozen component boundaries and trust boundaries, without historical phase narration.

Every external command is labeled with its required host/service/checkpoint.

- [x] **Step 3: Create the research index**

`research/README.md` contains:

1. research question and falsifiable hypothesis;
2. independent variables, controlled variables, dependent metrics, and hard constraints;
3. capability/claim matrix (`simulated`, `measured`, `blocked`, `not implemented`);
4. experiment entry-point table;
5. evidence and negative-result index;
6. current limitations and next valid experiment.

Do not describe engineering completeness as algorithmic novelty.

- [x] **Step 4: Update package metadata and check links**

Set:

```toml
description = "Device-aware research harness for reproducible MiniMax H3 optimization experiments"
```

Run a local Markdown-link checker implemented as a short Python one-liner using only `pathlib` and `re`; every relative link in `README.md` and active `docs/*.md` must resolve.

- [x] **Step 5: Run documentation-facing commands and commit**

Run:

```bash
.venv/bin/python -m harness4h3 --help
.venv/bin/python -m harness4h3 --config configs/default.yaml validate-config --json
.venv/bin/python -m harness4h3 validate --target configs/targets/rtx5080_example.yaml --json
git add README.md docs research/README.md pyproject.toml
git commit -m "docs: frame Harness4H3 as a reproducible research harness"
```

Expected: both offline validation commands exit `0`; README links resolve.

---

### Task 5: Archive completed design history and verify the full repository

**Files:**
- Move: completed `docs/superpowers/plans/*.md` to `research/history/plans/`
- Move: completed `docs/superpowers/specs/*.md` to `research/history/specs/`
- Modify: references to moved current design/plan if present

**Interfaces:**
- Consumes: Completed design and implementation records.
- Produces: No active planning clutter after implementation; all historical reasoning remains tracked.

- [x] **Step 1: Run the full verification suite before archiving the active plan**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q harness4h3 research tools tests
.venv/bin/python -m harness4h3 --help
.venv/bin/python -m harness4h3 optimize --target configs/targets/mobile_example.yaml --session-dir /tmp/harness4h3-reorg-smoke --session-id reorg-smoke --json
```

Expected: tests and compileall pass; help parses; fake optimization produces a valid bounded outcome (exit `0` only if its target is satisfied, otherwise its documented nonzero result is acceptable when JSON is valid).

- [x] **Step 2: Confirm moves did not mutate evidence or user M6 logic**

Run hash comparisons against `/tmp/harness4h3-evidence-before.sha256` and inspect:

```bash
git log --follow --oneline -- research/experiments/m6_campaign.py
git log --follow --oneline -- research/experiments/m6_runtime_recipe.py
git diff 6f640df -- research/experiments/m6_campaign.py research/experiments/m6_runtime_recipe.py
```

Expected: both histories follow across the rename and the user's additions remain visible.

- [x] **Step 3: Move completed plans/specifications into research history**

Run:

```bash
mkdir -p research/history/plans research/history/specs
git mv docs/superpowers/plans/*.md research/history/plans/
git mv docs/superpowers/specs/*.md research/history/specs/
```

Update the research index to point at `research/history/plans/` and `research/history/specs/`.

- [x] **Step 4: Run final cleanliness checks**

Run:

```bash
git diff --check
find . -name .DS_Store -not -path './.git/*'
git status --short
```

Expected: no whitespace errors, no `.DS_Store`, and only intended reorganization changes are staged/unstaged.

- [x] **Step 5: Commit the completed research archive**

Run:

```bash
git add docs/superpowers research/history research/README.md
git commit -m "docs: archive completed research design history"
```

- [x] **Step 6: Record final evidence**

Run:

```bash
git status --short
git log -5 --oneline
```

Expected: clean working tree and a commit sequence separating package movement, device preflight, evidence/history movement, research documentation, and final archive.
