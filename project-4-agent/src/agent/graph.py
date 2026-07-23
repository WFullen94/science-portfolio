"""Stage 2 — the LangGraph ReAct agent.

Wires the three tools to a local tool-calling LLM. The agent plans: it reads the
question, picks a tool, observes the result, and repeats until it can answer.
investigate() also returns the *trajectory* (which tools were called, in order) —
that's what the agent evaluation scores, not just the final text.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from agent.config import load_config
from agent.tools import ALL_TOOLS

SYSTEM = (
    "You are a cyber threat intelligence analyst assistant. Investigate the "
    "question using the provided tools, then give a concise final answer.\n"
    "Tool guidance:\n"
    "- cve_lookup: when the question names a specific CVE id (e.g. CVE-2021-44228).\n"
    "- attack_search: for 'how do adversaries...' or 'what technique...' questions.\n"
    "- map_indicator_to_technique: to classify ONE observed behavior into a "
    "single ATT&CK technique.\n"
    "Cite technique ids (T####) and CVE ids in your final answer."
)


def build_agent(cfg=None):
    cfg = cfg or load_config()
    llm = ChatOllama(model=cfg["agent"]["model"], temperature=cfg["agent"]["temperature"])
    return create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM)


@lru_cache(maxsize=1)
def get_agent():
    return build_agent()


def investigate(question: str) -> dict:
    cfg = load_config()
    recursion_limit = cfg["agent"]["max_iterations"] * 2 + 1
    result = get_agent().invoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": recursion_limit},
    )
    messages = result["messages"]
    # Trajectory: tool names in the order the agent called them.
    trajectory = [
        call["name"]
        for m in messages
        for call in (getattr(m, "tool_calls", None) or [])
    ]
    return {
        "question": question,
        "answer": messages[-1].content,
        "trajectory": trajectory,
        "n_steps": len(messages),
    }


def main() -> int:
    q = " ".join(sys.argv[1:]) or "What is CVE-2021-44228 and which ATT&CK technique would exploiting it enable?"
    out = investigate(q)
    print(f"Q: {out['question']}\n")
    print(f"TOOLS USED: {out['trajectory']}\n")
    print(f"ANSWER:\n{out['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
