"""Stage 2 — Feature engineering: bronze -> silver -> gold, all in PySpark.

silver: cleaned + typed flows (drop identifiers/leakage cols, cast, null-fill).
gold  : silver + engineered flow-behaviour features, split into train/test Delta
        tables ready for modeling.

The derived features encode NIDS domain knowledge (per-packet sizes, direction
ratios, throughput) that raw NetFlow counters don't express directly — the kind
of transform that justifies a distributed engine when the table is large.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from spine import schema
from spine.config import get_spark, load_config, resolve

# Columns that must never reach the model: raw identifiers (memorization /
# leakage risk) and ingest metadata.
META_COLUMNS = ["_ingest_source", "_ingest_ts"]


def to_silver(bronze: DataFrame, cfg) -> DataFrame:
    """Clean + type the raw flows."""
    drop = cfg["dataset"]["drop_columns"] + META_COLUMNS
    keep = [c for c in bronze.columns if c not in drop]
    df = bronze.select(*keep)

    # Cast all numeric features to double; coerce non-parseable to null then 0.
    for col in schema.NUMERIC_FEATURES:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast("double"))
    df = df.fillna(0.0, subset=[c for c in schema.NUMERIC_FEATURES if c in df.columns])

    # Labels: binary int, multiclass string.
    df = df.withColumn(schema.LABEL_BINARY, F.col(schema.LABEL_BINARY).cast("int"))
    df = df.withColumn(schema.LABEL_MULTICLASS, F.col(schema.LABEL_MULTICLASS).cast("string"))
    return df


def _safe_div(num, den):
    """Ratio with a guarded denominator (avoids div-by-zero -> inf/null)."""
    return num / F.when(den == 0, F.lit(1.0)).otherwise(den)


def engineer(silver: DataFrame) -> DataFrame:
    """Add NIDS behaviour features on top of the silver flows."""
    df = silver
    df = df.withColumn("TOTAL_BYTES", F.col("IN_BYTES") + F.col("OUT_BYTES"))
    df = df.withColumn("TOTAL_PKTS", F.col("IN_PKTS") + F.col("OUT_PKTS"))
    df = df.withColumn("BYTES_PER_PKT_IN", _safe_div(F.col("IN_BYTES"), F.col("IN_PKTS")))
    df = df.withColumn("BYTES_PER_PKT_OUT", _safe_div(F.col("OUT_BYTES"), F.col("OUT_PKTS")))
    df = df.withColumn("BYTE_DIR_RATIO", _safe_div(F.col("IN_BYTES"), F.col("OUT_BYTES") + F.lit(1.0)))
    df = df.withColumn("PKT_DIR_RATIO", _safe_div(F.col("IN_PKTS"), F.col("OUT_PKTS") + F.lit(1.0)))
    df = df.withColumn(
        "BYTES_PER_MS",
        _safe_div(F.col("TOTAL_BYTES"), F.col("FLOW_DURATION_MILLISECONDS") + F.lit(1.0)),
    )
    df = df.withColumn("PKT_SIZE_RANGE", F.col("MAX_IP_PKT_LEN") - F.col("MIN_IP_PKT_LEN"))
    df = df.withColumn("TTL_RANGE", F.col("MAX_TTL") - F.col("MIN_TTL"))
    df = df.withColumn(
        "RETRANS_RATE",
        _safe_div(
            F.col("RETRANSMITTED_IN_PKTS") + F.col("RETRANSMITTED_OUT_PKTS"),
            F.col("TOTAL_PKTS"),
        ),
    )
    return df


ENGINEERED_FEATURES = [
    "TOTAL_BYTES", "TOTAL_PKTS", "BYTES_PER_PKT_IN", "BYTES_PER_PKT_OUT",
    "BYTE_DIR_RATIO", "PKT_DIR_RATIO", "BYTES_PER_MS", "PKT_SIZE_RANGE",
    "TTL_RANGE", "RETRANS_RATE",
]

# Deterministic model feature order: raw NetFlow-v2 features then engineered.
# Matches feature_columns(gold) exactly (see README) so serving needs no Spark.
MODEL_FEATURES = schema.NUMERIC_FEATURES + ENGINEERED_FEATURES


def engineer_pandas(pdf):
    """pandas twin of engineer() — same 10 features, for Spark-free serving.

    Kept behaviourally identical to the Spark path; a parity test guards drift.
    """
    df = pdf.copy()

    def sdiv(num, den):
        den = den.mask(den == 0, 1.0)
        return num / den

    df["TOTAL_BYTES"] = df["IN_BYTES"] + df["OUT_BYTES"]
    df["TOTAL_PKTS"] = df["IN_PKTS"] + df["OUT_PKTS"]
    df["BYTES_PER_PKT_IN"] = sdiv(df["IN_BYTES"], df["IN_PKTS"])
    df["BYTES_PER_PKT_OUT"] = sdiv(df["OUT_BYTES"], df["OUT_PKTS"])
    df["BYTE_DIR_RATIO"] = sdiv(df["IN_BYTES"], df["OUT_BYTES"] + 1.0)
    df["PKT_DIR_RATIO"] = sdiv(df["IN_PKTS"], df["OUT_PKTS"] + 1.0)
    df["BYTES_PER_MS"] = sdiv(df["TOTAL_BYTES"], df["FLOW_DURATION_MILLISECONDS"] + 1.0)
    df["PKT_SIZE_RANGE"] = df["MAX_IP_PKT_LEN"] - df["MIN_IP_PKT_LEN"]
    df["TTL_RANGE"] = df["MAX_TTL"] - df["MIN_TTL"]
    df["RETRANS_RATE"] = sdiv(
        df["RETRANSMITTED_IN_PKTS"] + df["RETRANSMITTED_OUT_PKTS"], df["TOTAL_PKTS"]
    )
    return df


def feature_columns(df: DataFrame) -> list[str]:
    """Model input columns = everything except the labels."""
    return [c for c in df.columns if c not in (schema.LABEL_BINARY, schema.LABEL_MULTICLASS)]


def to_model_matrix(pdf, feats):
    """pandas frame -> float32 model matrix, used by both train and serve.

    Some raw NetFlow byte-rate columns (e.g. *_SECOND_BYTES) carry absurd finite
    values that overflow to +/-inf when cast to float32. We map inf -> NaN and
    let XGBoost handle NaN as missing, so train and inference stay consistent.
    """
    import numpy as np

    X = pdf[feats].astype("float32")
    return X.replace([np.inf, -np.inf], np.nan)


def main() -> int:
    cfg = load_config()
    spark = get_spark("spine-features")
    try:
        paths = cfg["paths"]
        bronze = spark.read.format("delta").load(str(resolve(paths["bronze"])))
        print(f"[features] bronze rows: {bronze.count():,}")

        silver = to_silver(bronze, cfg)
        silver.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).save(str(resolve(paths["silver"])))
        print(f"[features] wrote silver -> {resolve(paths['silver'])}")

        gold = engineer(silver)

        test_size = cfg["train"]["test_size"]
        seed = cfg["train"]["random_state"]
        train_df, test_df = gold.randomSplit([1.0 - test_size, test_size], seed=seed)

        train_df.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).save(str(resolve(paths["gold_train"])))
        test_df.write.format("delta").mode("overwrite").option(
            "overwriteSchema", "true"
        ).save(str(resolve(paths["gold_test"])))

        n_feat = len(feature_columns(gold))
        print(f"[features] wrote gold train/test ({n_feat} features)")
        print(f"[features]   train: {train_df.count():,}   test: {test_df.count():,}")
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
