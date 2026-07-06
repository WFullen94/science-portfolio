"""Feature-engineering transform tests (PySpark)."""

from spine import schema
from spine.features import ENGINEERED_FEATURES, engineer, feature_columns


def _tiny_silver(spark):
    # Two flows: one normal, one with a zero-packet denominator to exercise the
    # safe-division guard.
    rows = [
        # IN_BYTES, IN_PKTS, OUT_BYTES, OUT_PKTS, FLOW_DURATION_MS, MAX/MIN len/ttl, retrans, Label, Attack
        (1000.0, 10.0, 500.0, 5.0, 200.0, 1500.0, 40.0, 64.0, 32.0, 1.0, 1.0, 1, "scanning"),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, "Benign"),
    ]
    cols = [
        "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS",
        "MAX_IP_PKT_LEN", "MIN_IP_PKT_LEN", "MAX_TTL", "MIN_TTL",
        "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_PKTS",
        schema.LABEL_BINARY, schema.LABEL_MULTICLASS,
    ]
    return spark.createDataFrame(rows, cols)


def test_engineered_features_present_and_finite(spark):
    gold = engineer(_tiny_silver(spark))
    for col in ENGINEERED_FEATURES:
        assert col in gold.columns

    rows = [r.asDict() for r in gold.collect()]
    # No NaN/inf leaked from the zero-denominator row.
    for row in rows:
        for col in ENGINEERED_FEATURES:
            v = row[col]
            assert v is not None
            assert v == v            # not NaN
            assert abs(v) != float("inf")

    # Sanity on the normal flow's arithmetic.
    normal = next(r for r in rows if r[schema.LABEL_BINARY] == 1)
    assert normal["TOTAL_BYTES"] == 1500.0
    assert normal["BYTES_PER_PKT_IN"] == 100.0


def test_feature_columns_excludes_labels(spark):
    gold = engineer(_tiny_silver(spark))
    feats = feature_columns(gold)
    assert schema.LABEL_BINARY not in feats
    assert schema.LABEL_MULTICLASS not in feats
