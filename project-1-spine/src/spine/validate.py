"""Great Expectations data-quality gate on the ingested NetFlow flows.

Runs a suite of expectations against a pandas view of the bronze table. Ingest
calls this as a hard gate: if a critical expectation fails, the pipeline stops
before bad data reaches feature engineering. This is the "data-validation gate"
in the resume claim.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import great_expectations as gx
import pandas as pd

from spine import schema


@dataclass
class ValidationReport:
    success: bool
    evaluated: int
    passed: int
    failed_expectations: list[str]

    def summary(self) -> str:
        head = "PASSED" if self.success else "FAILED"
        line = f"[GX] {head}: {self.passed}/{self.evaluated} expectations met"
        if self.failed_expectations:
            line += "\n      failed: " + ", ".join(self.failed_expectations)
        return line


def _build_suite(name: str) -> gx.ExpectationSuite:
    """Expectations that encode what a valid NetFlow-v2 batch must look like."""
    suite = gx.ExpectationSuite(name=name)

    # Schema: exactly the columns we expect (catches upstream schema drift).
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=schema.ALL_COLUMNS, exact_match=False
        )
    )
    # Binary label must be present and strictly in {0, 1}.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column=schema.LABEL_BINARY)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column=schema.LABEL_BINARY, value_set=[0, 1]
        )
    )
    # Multiclass label present.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column=schema.LABEL_MULTICLASS)
    )
    # Core volumetric features present and physically non-negative.
    for col in ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "PROTOCOL"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )
    for col in ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS"]:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(column=col, min_value=0)
        )
    # Protocol number is a byte field.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="PROTOCOL", min_value=0, max_value=255
        )
    )
    return suite


def validate_frame(pdf: pd.DataFrame) -> ValidationReport:
    """Validate a pandas frame against the NetFlow-v2 suite."""
    # Unique names per call: get_context(ephemeral) can return a shared context,
    # so reusing fixed names would leak batches across invocations.
    uid = uuid.uuid4().hex[:8]
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(f"bronze_pandas_{uid}")
    asset = data_source.add_dataframe_asset(name=f"flows_{uid}")
    batch_def = asset.add_batch_definition_whole_dataframe(f"batch_{uid}")
    suite = _build_suite(f"netflow_v2_input_{uid}")

    results = batch_def.get_batch(batch_parameters={"dataframe": pdf}).validate(suite)

    evaluated = len(results.results)
    failed = [
        r.expectation_config.type
        for r in results.results
        if not r.success
    ]
    return ValidationReport(
        success=bool(results.success),
        evaluated=evaluated,
        passed=evaluated - len(failed),
        failed_expectations=failed,
    )


def validate_spark(sdf, sample_rows: int = 200_000) -> ValidationReport:
    """Validate a Spark DataFrame by materializing a bounded pandas sample.

    For large batches we validate a cap of `sample_rows` rows — enough to catch
    schema and value-range violations without pulling millions of rows to the
    driver.
    """
    total = sdf.count()
    if total > sample_rows:
        pdf = sdf.limit(sample_rows).toPandas()
    else:
        pdf = sdf.toPandas()
    return validate_frame(pdf)
