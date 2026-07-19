"""Stage 5 (orchestration) — Evidently data-drift check + retrain decision.

Compares a reference feature distribution (the training data) against the current
one (freshly ingested flows). If the share of drifted columns crosses the
configured threshold, it emits a retrain signal that the Airflow DAG branches on.

Writes:
  data/reports/drift_result.json   machine-readable signal (used by the DAG)
  data/reports/drift_report.html   human-readable Evidently report

Set SPINE_DRIFT_SIMULATE=1 to perturb the current data and force a drift, for
demonstrating the trigger end to end.
"""

from __future__ import annotations

import json
import os

from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from spine.config import get_spark, load_config, resolve
from spine.features import MODEL_FEATURES

# Airflow branch task ids (kept here so the DAG imports no heavy deps to decide).
RETRAIN_TASK = "retrain_features"
SKIP_TASK = "skip_retrain"


def _load_samples(cfg, n: int = 15_000):
    spark = get_spark("spine-drift")
    try:
        paths = cfg["paths"]
        train = spark.read.format("delta").load(str(resolve(paths["gold_train"])))
        test = spark.read.format("delta").load(str(resolve(paths["gold_test"])))
        cols = [c for c in MODEL_FEATURES if c in train.columns]
        ref = train.select(*cols).limit(n).toPandas()
        cur = test.select(*cols).limit(n).toPandas()
    finally:
        spark.stop()
    return ref, cur


def _simulate_drift(cur):
    """Shift/scale numeric features to fabricate distribution drift (demo only)."""
    perturbed = cur.copy()
    for col in perturbed.columns[: max(1, len(perturbed.columns) // 2)]:
        perturbed[col] = perturbed[col] * 3.0 + perturbed[col].std()
    return perturbed


def _extract_drift(report_dict) -> tuple[float, int, int]:
    """Pull (share_of_drifted_columns, n_drifted, n_columns) from Evidently."""
    for metric in report_dict.get("metrics", []):
        res = metric.get("result", {})
        if "share_of_drifted_columns" in res:
            return (
                float(res["share_of_drifted_columns"]),
                int(res.get("number_of_drifted_columns", 0)),
                int(res.get("number_of_columns", 0)),
            )
    raise RuntimeError("Evidently result did not contain drift share")


def run_drift_check(cfg) -> dict:
    ref, cur = _load_samples(cfg)
    if os.environ.get("SPINE_DRIFT_SIMULATE") == "1":
        cur = _simulate_drift(cur)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    share, n_drifted, n_cols = _extract_drift(report.as_dict())

    threshold = cfg["drift"]["dataset_drift_threshold"]
    drift_detected = share >= threshold

    reports_dir = resolve(cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report.save_html(str(reports_dir / "drift_report.html"))

    result = {
        "drift_detected": bool(drift_detected),
        "share_of_drifted_columns": round(share, 4),
        "n_drifted_columns": n_drifted,
        "n_columns": n_cols,
        "threshold": threshold,
        "decision": RETRAIN_TASK if drift_detected else SKIP_TASK,
    }
    (reports_dir / "drift_result.json").write_text(json.dumps(result, indent=2))
    print(f"[drift] share={share:.3f} (>= {threshold}? {drift_detected}) "
          f"-> {result['decision']}")
    return result


def decide_branch() -> str:
    """Airflow branch callable: read the last drift result, return next task id."""
    cfg = load_config()
    result_path = resolve(cfg["paths"]["reports"]) / "drift_result.json"
    if not result_path.exists():
        return SKIP_TASK
    return json.loads(result_path.read_text()).get("decision", SKIP_TASK)


def main() -> int:
    run_drift_check(load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
