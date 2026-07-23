"""Stage 3 — Agent evaluation: score the agent as an agent, not just its answer.

For each curated task we run the agent and measure three things most people skip:
  * tool-selection accuracy — did it call the tool the task actually needed?
  * trajectory validity      — a sane path (>=1 tool, no pathological looping)?
  * task completion          — does the final answer contain the expected id?

These are computed from the agent's trajectory (the tools it called, in order),
which is exactly what distinguishes agent evaluation from plain answer scoring.
"""

from __future__ import annotations

import json
from collections import Counter

from agent.config import load_config, resolve
from agent.graph import investigate


def load_tasks(cfg) -> list[dict]:
    rows = []
    with open(resolve(cfg["paths"]["eval_set"])) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_task(task: dict, out: dict, max_iters: int) -> dict:
    traj = out["trajectory"]
    answer = out["answer"].lower()
    counts = Counter(traj)
    return {
        "question": task["question"],
        "expected_tool": task["expected_tool"],
        "trajectory": traj,
        "tool_selection_ok": task["expected_tool"] in traj,
        "task_completed": all(s.lower() in answer for s in task["expected_contains"]),
        # Valid = it used at least one tool, didn't exceed the loop budget, and
        # didn't call the same tool more than twice (no pathological looping).
        "trajectory_valid": bool(traj) and len(traj) <= max_iters
        and (max(counts.values(), default=0) <= 2),
    }


def main() -> int:
    cfg = load_config()
    tasks = load_tasks(cfg)
    max_iters = cfg["agent"]["max_iterations"]
    print(f"[agent-eval] running the agent on {len(tasks)} tasks ...")

    rows = []
    for i, task in enumerate(tasks, 1):
        out = investigate(task["question"])
        row = score_task(task, out, max_iters)
        rows.append(row)
        print(f"[agent-eval]   {i}/{len(tasks)} tool={row['trajectory']} "
              f"select={'Y' if row['tool_selection_ok'] else 'N'} "
              f"done={'Y' if row['task_completed'] else 'N'}")

    n = len(rows)
    summary = {
        "n_tasks": n,
        "tool_selection_accuracy": round(sum(r["tool_selection_ok"] for r in rows) / n, 3),
        "trajectory_validity_rate": round(sum(r["trajectory_valid"] for r in rows) / n, 3),
        "task_completion_rate": round(sum(r["task_completed"] for r in rows) / n, 3),
    }

    reports = resolve(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "agent_eval.json").write_text(
        json.dumps({"summary": summary, "tasks": rows}, indent=2)
    )
    print("\n[agent-eval] === summary ===")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    print(f"[agent-eval] wrote {reports/'agent_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
