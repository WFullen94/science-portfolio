"""Stage 4 — Ragas evaluation: measure the RAG system, don't just eyeball it.

Runs the RAG chain over a curated question set, then scores it with Ragas:
  * Faithfulness              — is the answer grounded in the retrieved context?  (generator)
  * ResponseRelevancy         — does the answer address the question?             (generator)
  * ContextPrecisionWithRef   — are relevant chunks retrieved and ranked high?    (retriever)
  * ContextRecall             — did retrieval fetch everything the reference needs? (retriever)

Splitting retriever vs generator metrics is the point: it tells you which
component to fix. The LLM judge is the same pluggable backend as generation.
"""

from __future__ import annotations

import json

from rag.chain import build_llm, get_chain
from rag.config import load_config, resolve
from rag.index import build_embeddings


def load_eval(cfg) -> list[dict]:
    rows = []
    with open(resolve(cfg["paths"]["eval_set"])) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(cfg):
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    chain = get_chain()
    rows = load_eval(cfg)
    print(f"[eval] running the RAG chain over {len(rows)} questions ...")
    samples = []
    for i, row in enumerate(rows, 1):
        out = chain.invoke(row["user_input"])
        samples.append(SingleTurnSample(
            user_input=row["user_input"],
            response=out["answer"],
            retrieved_contexts=out["contexts"],
            reference=row["reference"],
        ))
        print(f"[eval]   answered {i}/{len(rows)}")

    dataset = EvaluationDataset(samples=samples)
    judge = LangchainLLMWrapper(build_llm(cfg))
    embeddings = LangchainEmbeddingsWrapper(build_embeddings(cfg))
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    print("[eval] scoring with Ragas (local LLM judge — this is the slow part) ...")
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=3, timeout=300),
    )


def main() -> int:
    cfg = load_config()
    result = run(cfg)
    df = result.to_pandas()

    reports = resolve(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports / "ragas_details.csv", index=False)

    # Aggregate each metric (mean over the numeric score columns, skipping NaN
    # rows where the local judge failed to return a parseable score).
    summary = {}
    print("\n[eval] === Ragas scores (mean over eval set) ===")
    for col in df.columns:
        if df[col].dtype.kind in "fc":
            n_ok = int(df[col].notna().sum())
            mean = float(df[col].mean(skipna=True))
            summary[col] = {"mean": round(mean, 4), "n_scored": n_ok, "n_total": len(df)}
            print(f"   {col:42s} {mean:.4f}   ({n_ok}/{len(df)} scored)")
    (reports / "ragas_scores.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[eval] wrote {reports/'ragas_scores.json'} and ragas_details.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
