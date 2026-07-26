"""Stage 1 — Build the CTI -> ATT&CK classification dataset from STIX procedures.

MITRE's "uses" relationships carry a free-text *procedure* description of how a
group or piece of software applies a technique — e.g. "[APT29] has used Cobalt
Strike to inject into processes". The description is the input text; the target
technique is the label. We clean the STIX markdown/citations, keep the top-K most
frequent techniques (so classes have enough examples), and make a stratified split.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from ctilora.config import load_config, resolve

_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")     # [text](url) -> text
_CITATION = re.compile(r"\(Citation:[^)]*\)")     # remove (Citation: ...)
_WS = re.compile(r"\s+")


def download_stix(cfg) -> dict:
    cache = resolve(cfg["corpus"]["cache"])
    if cache.exists():
        return json.loads(cache.read_text())
    cache.parent.mkdir(parents=True, exist_ok=True)
    data = requests.get(cfg["corpus"]["stix_url"], timeout=180).json()
    cache.write_text(json.dumps(data))
    return data


def _clean(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = _CITATION.sub("", text)
    return _WS.sub(" ", text).strip()


def _ext_id(obj: dict) -> str:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "")
    return ""


def build_examples(data: dict) -> pd.DataFrame:
    # technique STIX id -> (external id, name)
    tech = {}
    for obj in data["objects"]:
        if obj.get("type") == "attack-pattern" and not obj.get("revoked") \
                and not obj.get("x_mitre_deprecated"):
            eid = _ext_id(obj)
            if eid:
                tech[obj["id"]] = (eid, obj.get("name", ""))

    rows = []
    for obj in data["objects"]:
        if obj.get("type") != "relationship" or obj.get("relationship_type") != "uses":
            continue
        target = obj.get("target_ref", "")
        desc = obj.get("description")
        if target in tech and desc:
            text = _clean(desc)
            if len(text) >= 20:  # skip near-empty procedures
                eid, name = tech[target]
                rows.append({"text": text, "technique_id": eid, "technique_name": name})
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    dcfg = cfg["dataset"]
    df = build_examples(download_stix(cfg))

    # Keep the top-K techniques that clear the min-examples bar.
    counts = df["technique_id"].value_counts()
    keep = counts[counts >= dcfg["min_examples"]].head(dcfg["top_k_techniques"]).index
    df = df[df["technique_id"].isin(keep)].reset_index(drop=True)

    labels = sorted(df["technique_id"].unique())
    label_to_id = {t: i for i, t in enumerate(labels)}
    df["label"] = df["technique_id"].map(label_to_id)

    train_df, test_df = train_test_split(
        df, test_size=dcfg["test_size"], random_state=dcfg["seed"], stratify=df["label"]
    )
    df["split"] = "train"
    df.loc[test_df.index, "split"] = "test"

    data_path = resolve(cfg["paths"]["data"])
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

    id_to_name = dict(zip(df["technique_id"], df["technique_name"]))
    label_map = {str(label_to_id[t]): {"technique_id": t, "name": id_to_name[t]} for t in labels}
    resolve(cfg["paths"]["label_map"]).write_text(json.dumps(label_map, indent=2))

    print(f"[data] {len(df)} procedures across {len(labels)} techniques "
          f"({(df['split'] == 'train').sum()} train / {(df['split'] == 'test').sum()} test)")
    print(f"[data] examples per class: min {counts[keep].min()}, max {counts[keep].max()}")
    print(f"[data] sample: [{df.iloc[0]['technique_id']}] {df.iloc[0]['text'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
