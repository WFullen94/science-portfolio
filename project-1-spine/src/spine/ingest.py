"""Stage 1 — Ingest: download NetFlow flows and land them in a bronze Delta table.

Two acquisition paths, chosen by `dataset.max_rows` in config:
  * capped (dev)  — stream the first N rows over HTTP with pandas (no full pull)
  * full (scale)  — download the file once and read it with Spark

Either way we land an immutable, versioned bronze Delta table, then run the
Great Expectations gate before anything downstream may touch the data.
"""

from __future__ import annotations

import sys

import pandas as pd
from pyspark.sql import functions as F

from spine import validate
from spine.config import get_spark, load_config, resolve
from spine.schema import make_synthetic

HF_RESOLVE = "https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"


def _stream_csv(repo_id: str, filename: str, max_rows: int | None) -> pd.DataFrame:
    url = HF_RESOLVE.format(repo_id=repo_id, filename=filename)
    print(f"[ingest] streaming {max_rows or 'ALL'} rows from {repo_id}/{filename}")
    return pd.read_csv(url, nrows=max_rows)


def _download_full(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    print(f"[ingest] downloading full file {repo_id}/{filename} (cached)")
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")


def acquire(spark, cfg):
    """Return (spark_df, provenance_str), trying each source then synthetic."""
    ds = cfg["dataset"]
    max_rows = ds.get("max_rows")

    for src in ds.get("sources", []):
        repo_id, filename = src["repo_id"], src["filename"]
        try:
            if max_rows is not None:
                pdf = _stream_csv(repo_id, filename, max_rows)
                sdf = spark.createDataFrame(pdf)
            else:
                local = _download_full(repo_id, filename)
                sdf = spark.read.csv(local, header=True, inferSchema=True)
            return sdf, f"hf:{repo_id}/{filename}"
        except Exception as exc:  # network / repo / parse issues -> next source
            print(f"[ingest] source {repo_id} failed: {exc!r}")

    # Offline fallback: schema-accurate synthetic data so the pipeline still runs.
    print("[ingest] all remote sources failed — generating synthetic fallback")
    pdf = make_synthetic(n_rows=ds.get("synthetic_fallback_rows", 60_000))
    return spark.createDataFrame(pdf), "synthetic-fallback"


def write_bronze(sdf, provenance: str, cfg):
    """Land raw flows as a bronze Delta table, tagged with ingest provenance."""
    bronze_path = str(resolve(cfg["paths"]["bronze"]))
    sdf = (
        sdf.withColumn("_ingest_source", F.lit(provenance))
        .withColumn("_ingest_ts", F.current_timestamp())
    )
    (
        # Coalesce to a few writers: fewer small files and lower local-heap
        # pressure than the default per-core partitioning.
        sdf.coalesce(4)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(bronze_path)
    )
    print(f"[ingest] wrote bronze Delta -> {bronze_path}")
    return bronze_path


def main() -> int:
    cfg = load_config()
    spark = get_spark("spine-ingest")
    try:
        sdf, provenance = acquire(spark, cfg)
        n = sdf.count()
        print(f"[ingest] acquired {n:,} flows via {provenance}")

        bronze_path = write_bronze(sdf, provenance, cfg)
        bronze = spark.read.format("delta").load(bronze_path)

        # --- Great Expectations gate ---
        report = validate.validate_spark(bronze)
        print(report.summary())
        if not report.success:
            print("[ingest] ABORT: input data failed validation gate", file=sys.stderr)
            return 1

        # Quick label sanity readout.
        print("[ingest] label distribution:")
        bronze.groupBy("Label").count().orderBy("Label").show()
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
