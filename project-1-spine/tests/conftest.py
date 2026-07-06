"""Shared pytest fixtures. The Spark session is session-scoped so the JVM
starts once for the whole test run."""

import pytest


@pytest.fixture(scope="session")
def spark():
    from spine.config import get_spark

    s = get_spark("spine-tests", shuffle_partitions=2)
    yield s
    s.stop()
