"""Validation-gate tests — no Spark needed, run in seconds."""

from spine import schema
from spine.validate import validate_frame


def test_synthetic_sample_passes_gate():
    df = schema.make_synthetic(n_rows=2_000, seed=0)
    report = validate_frame(df)
    assert report.success, report.failed_expectations
    assert report.evaluated == report.passed


def test_out_of_range_label_fails_gate():
    df = schema.make_synthetic(n_rows=2_000, seed=0)
    df.loc[0, schema.LABEL_BINARY] = 7  # illegal label value
    report = validate_frame(df)
    assert not report.success
    assert any("in_set" in e for e in report.failed_expectations)


def test_negative_bytes_fail_gate():
    df = schema.make_synthetic(n_rows=2_000, seed=0)
    df.loc[0, "IN_BYTES"] = -5  # bytes can't be negative
    report = validate_frame(df)
    assert not report.success
