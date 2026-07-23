# Project 4 — Threat-Investigation Agent + Agent Evaluation

A multi-tool agent that investigates threat-intel questions by planning and calling tools —
then, the part that matters, is **evaluated as an agent**: not just final-answer quality, but
**tool-selection accuracy** and **trajectory validity**.

```
question -> LangGraph agent (llama3.1:8b) --plans--> picks a tool --> observes --> ... -> answer
   tools:  attack_search (ATT&CK retriever, reused from P3)
           cve_lookup (live NVD API)
           map_indicator_to_technique (behavior -> ATT&CK technique)
```

Builds directly on **Project 3**: the ATT&CK retriever becomes *one tool among several*. Mirrors
how real CTI pipelines link CVE ↔ ATT&CK ↔ observed behavior.

## Stack

| Piece | Tool |
|-------|------|
| Orchestration | **LangGraph** (ReAct-style plan/act loop) |
| Tools / function calling | LangChain tools, structured args |
| LLM | local **Ollama** llama3.1:8b (verified to do tool calling) |
| Agent evaluation | tool-selection accuracy, trajectory validity, task completion |
| Tracing | **Phoenix** (shared with P3) |

## Environment

Shares Project 3's environment (same LLM + retriever). From `project-3-rag/.venv`:

```bash
pip install -r ../project-4-agent/requirements.txt
pip install -e ../project-3-rag ../project-4-agent
```

## Evaluation results (local llama3.1:8b, 8 curated tasks)

| Metric | Score |
|--------|-------|
| Tool-selection accuracy | **1.00** — always chose the right tool |
| Trajectory validity | **1.00** — clean single-tool paths, no looping |
| Task completion | **0.875** — one miss, traced to the mapping *tool's* retrieval, not the agent's planning |

That decomposition is the point: agent eval separates *planning* quality from *tool* quality.
Run `python -m agent.evaluate`; per-task detail lands in `data/reports/agent_eval.json`.

## Tracing

`AGENT_TRACING=1` turns on Phoenix. A single run produces the full trajectory as a span tree —
`LangGraph → agent → call_model (ChatOllama) → should_continue → tools → <tool> → …` — so you can
see exactly which tool the agent picked and what it observed.

## Layout

```
conf/config.yaml        model, NVD API, eval paths
src/agent/
  tools.py              the 3 tools (ATT&CK search, NVD CVE lookup, technique mapping)
  graph.py              LangGraph ReAct agent + investigate() (returns trajectory)
  evaluate.py           agent eval: tool-selection + trajectory + completion
  tracing.py            Phoenix instrumentation (shared with P3)
eval/agent_tasks.jsonl  curated tasks with expected tool + answer
```
