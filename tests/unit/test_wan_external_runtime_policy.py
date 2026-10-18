from __future__ import annotations

import pytest

from harness4h3.operators.wan.runtime_policy import MagCachePolicy, SVOOPolicy
from harness4h3.operators.wan.search_space import default_search_space


def test_external_runtime_policies_require_method_and_preserve_upstream_fields():
    with pytest.raises(ValueError, match="method must be 'svoo'"):
        SVOOPolicy.from_mapping({"config": {}})
    with pytest.raises(ValueError, match="method must be 'magcache'"):
        MagCachePolicy.from_mapping({"config": {}})

    svoo = SVOOPolicy.from_mapping(
        {"method": "svoo", "model_key": "wan21-t2v-1.3b", "config": {"future_field": 7}}
    )
    cache = MagCachePolicy.from_mapping({"method": "magcache", "config": {"future_flag": True}})
    assert svoo.to_dict()["config"] == {"future_field": 7}
    assert cache.to_dict()["config"] == {"future_flag": True}


def test_default_search_catalog_keeps_upstream_methods_distinct():
    operators = {item.name: item for item in default_search_space().operators}
    assert operators["sparse_runtime"].granularity == "sparse_runtime"
    assert operators["svoo_attention"].implementation_ref.endswith("SVOOAttentionOperator")
    assert operators["magcache"].implementation_ref.endswith("MagCacheOperator")
