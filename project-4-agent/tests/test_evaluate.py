"""Agent-eval scoring-logic tests (pure functions, no LLM)."""

from agent.evaluate import score_task


def test_correct_run_scores_all_true():
    task = {"question": "q", "expected_tool": "cve_lookup",
            "expected_contains": ["CVE-2021-44228", "10.0"]}
    out = {"trajectory": ["cve_lookup"], "answer": "CVE-2021-44228 has CVSS 10.0"}
    r = score_task(task, out, max_iters=6)
    assert r["tool_selection_ok"] and r["task_completed"] and r["trajectory_valid"]


def test_wrong_tool_and_missing_answer():
    task = {"question": "q", "expected_tool": "cve_lookup", "expected_contains": ["T1003"]}
    out = {"trajectory": ["attack_search"], "answer": "no technique id here"}
    r = score_task(task, out, max_iters=6)
    assert not r["tool_selection_ok"]
    assert not r["task_completed"]


def test_looping_trajectory_is_invalid():
    task = {"question": "q", "expected_tool": "attack_search", "expected_contains": []}
    out = {"trajectory": ["attack_search"] * 5, "answer": "x"}  # same tool >2x
    r = score_task(task, out, max_iters=6)
    assert not r["trajectory_valid"]


def test_empty_trajectory_is_invalid():
    task = {"question": "q", "expected_tool": "attack_search", "expected_contains": []}
    out = {"trajectory": [], "answer": "answered with no tools"}
    r = score_task(task, out, max_iters=6)
    assert not r["trajectory_valid"]
