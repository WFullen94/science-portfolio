"""Agent construction tests (no LLM server call)."""

from agent.graph import SYSTEM, build_agent


def test_agent_compiles():
    a = build_agent()
    assert hasattr(a, "invoke")


def test_system_prompt_names_every_tool():
    for name in ("cve_lookup", "attack_search", "map_indicator_to_technique"):
        assert name in SYSTEM
