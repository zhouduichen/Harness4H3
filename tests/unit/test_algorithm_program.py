from harness4h3.sandbox import AlgorithmSandbox


def test_program_entrypoint_receives_search_state_and_returns_raw_json(tmp_path):
    source = """def propose(profile, history):
    return {'operator': 'prune_blocks', 'args': {'block_ids': ['block.0']}}

def search(profile, history, state, search_space):
    return {
        'candidates': [
            {'operator': 'prune_blocks', 'args': {'block_ids': [state['block_id']]}}
        ],
        'rank': [0],
        'stop': True,
    }
"""
    result = AlgorithmSandbox(tmp_path).run_program(
        source,
        entrypoint="search",
        payload={"profile": {}, "history": [], "state": {"block_id": "block.2"}, "search_space": {"operators": []}},
    )
    assert result.status == "success"
    assert result.value["candidates"][0]["args"]["block_ids"] == ["block.2"]
    assert result.metadata["entrypoint"] == "search"


def test_program_entrypoint_rejects_source_without_requested_function(tmp_path):
    source = "def propose(profile, history):\n    return {}\n"
    result = AlgorithmSandbox(tmp_path).run_program(source, entrypoint="search", payload={})
    assert result.status == "failed"
    assert result.failure_type == "sandbox_policy"
    assert "search" in result.message

