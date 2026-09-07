from harness4h3.tools.registry import Tool, ToolRegistry


def test_tool_registry_exposes_schema_and_wraps_results():
    registry = ToolRegistry()
    registry.register(Tool("double", "Double a number", {"value": "number"}, lambda args: args["value"] * 2))
    assert registry.visible()[0]["name"] == "double"
    assert registry.execute("double", {"value": 3}).value == 6
    assert registry.execute("missing", {}).ok is False

