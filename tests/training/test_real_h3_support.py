import torch
from safetensors.torch import load_file, save_file

from tools.h3_real_support import patch_safetensors_tensors


def test_patch_safetensors_preserves_parent_and_replaces_fixed_ranges(tmp_path):
    parent = tmp_path / "parent.safetensors"
    child = tmp_path / "child.safetensors"
    original = {"head.weight": torch.zeros(4, dtype=torch.bfloat16), "frozen": torch.ones(2)}
    save_file(original, str(parent))

    patch_safetensors_tensors(parent, child, {"head.weight": torch.full((4,), 2, dtype=torch.bfloat16)})

    assert torch.equal(load_file(str(parent))["head.weight"], original["head.weight"])
    assert torch.equal(load_file(str(parent))["frozen"], original["frozen"])
    assert torch.equal(load_file(str(child))["head.weight"], torch.full((4,), 2, dtype=torch.bfloat16))
    assert torch.equal(load_file(str(child))["frozen"], original["frozen"])
