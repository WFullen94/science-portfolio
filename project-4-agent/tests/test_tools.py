"""Structural tests for the agent tools (no network / models needed)."""

from agent.tools import ALL_TOOLS


def test_three_distinct_tools_with_descriptions():
    names = [t.name for t in ALL_TOOLS]
    assert set(names) == {"attack_search", "cve_lookup", "map_indicator_to_technique"}
    for t in ALL_TOOLS:
        # A clear description is what lets the LLM pick the right tool.
        assert len(t.description) > 40


def test_tools_declare_one_string_arg():
    for t in ALL_TOOLS:
        schema = t.args
        assert len(schema) == 1  # each tool takes a single string argument
