def propose(profile, history):
    records = [item for item in profile.get("block_sensitivity", []) if item.get("available")]
    selected = min(records, key=lambda item: item["importance_score"])["block_id"] if records else profile["layers"][0]["block_id"]
    return {"operator": "prune_blocks", "args": {"block_ids": [selected]}}
