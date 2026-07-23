"""The agent's tools — three distinct capabilities it must choose between.

  attack_search            : semantic Q&A over MITRE ATT&CK (reuses the P3 retriever)
  cve_lookup               : real NVD API lookup of a specific CVE
  map_indicator_to_technique : classify an observed behavior -> single ATT&CK technique

Keeping the tools genuinely distinct is what makes tool-selection evaluation
meaningful: the agent has to pick the right one for each question.
"""

from __future__ import annotations

import requests
from langchain_core.tools import tool

from agent.config import load_config
from rag.retrieve import get_retriever  # P3 package, reused as a dependency


@tool
def attack_search(query: str) -> str:
    """Search the MITRE ATT&CK knowledge base for techniques matching a
    natural-language question about how adversaries operate. Use for 'how do
    adversaries...' or 'what technique covers...' questions. Returns technique
    ids, names, and descriptions."""
    docs = get_retriever().retrieve(query)
    if not docs:
        return "No matching ATT&CK techniques found."
    return "\n\n".join(
        f"{d.metadata['technique_id']} — {d.metadata['name']}: {d.page_content[:280]}"
        for d in docs
    )


@tool
def cve_lookup(cve_id: str) -> str:
    """Look up a specific CVE by its identifier (e.g. CVE-2021-44228) in the
    NVD. Use when the question references a concrete CVE id. Returns the
    description and CVSS severity."""
    cfg = load_config()["nvd"]
    try:
        resp = requests.get(cfg["api_url"], params={"cveId": cve_id.strip()},
                            timeout=cfg["timeout"])
        resp.raise_for_status()
        vulns = resp.json().get("vulnerabilities", [])
        if not vulns:
            return f"No NVD record found for {cve_id}."
        cve = vulns[0]["cve"]
        desc = next((d["value"] for d in cve.get("descriptions", [])
                     if d.get("lang") == "en"), "(no description)")
        severity = "unknown"
        metrics = cve.get("metrics", {})
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if metrics.get(key):
                data = metrics[key][0]["cvssData"]
                severity = f"{data.get('baseScore')} ({data.get('baseSeverity', 'n/a')})"
                break
        return f"{cve_id}: CVSS {severity}. {desc}"
    except requests.RequestException as exc:
        return f"NVD lookup failed for {cve_id}: {exc}"


@tool
def map_indicator_to_technique(indicator: str) -> str:
    """Given a single observed indicator or behavior (e.g. 'a scheduled task was
    created that runs at logon'), classify it to the single most likely MITRE
    ATT&CK technique. Returns one technique id and name. Use for mapping an
    observation to a technique, not for open-ended questions."""
    docs = get_retriever().retrieve(indicator)
    if not docs:
        return "No technique match."
    top = docs[0]
    return f"{top.metadata['technique_id']} — {top.metadata['name']}"


ALL_TOOLS = [attack_search, cve_lookup, map_indicator_to_technique]
