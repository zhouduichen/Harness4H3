from __future__ import annotations

from harness4h3.memory.design_gene import DesignGene, DesignGeneStore


def test_design_gene_store_is_append_only_and_round_trips(tmp_path):
    store = DesignGeneStore(tmp_path / "genes.jsonl")
    gene = DesignGene(
        gene_id="H3-NVFP4-Quantization-001",
        status="validated_sanity",
        state={"parent": "M0000", "child": "M0001"},
        bottleneck={"model_size": "high", "latency": "high"},
        intervention={"operator": "quantization", "scheme": "nvfp4"},
        controlled_conditions={"sampling_steps": 20, "seed": 42, "cache_reset_before_each_run": True},
        benefit={"model_size_reduction": 0.4026, "latency_reduction": 0.5642},
        remaining_limitation={"peak_memory_gb": 16.358, "target_limit_gb": 16.0},
        risks_lessons=["cache contamination can mimic a black-frame failure"],
        evidence={"experiment": "m5-final-controlled-20steps"},
    )
    store.append(gene)
    store.append(gene)
    records = list(store.read())
    assert len(records) == 2
    assert records[0].gene_id == gene.gene_id
    assert records[0].created_at
