"""Central config + Spark/Delta session for Project 1 (The Spine).

Every stage imports from here so paths, dataset schema, and the Delta-enabled
Spark session are defined in exactly one place.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repo layout: this file lives at project-1-spine/src/spine/config.py
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parents[1]  # project-1-spine/
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yaml"


@lru_cache(maxsize=None)
def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load and cache the YAML config."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r") as fh:
        return yaml.safe_load(fh)


def resolve(rel: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return (PROJECT_ROOT / rel).resolve()


def _ensure_java_home() -> None:
    """PySpark needs JAVA_HOME. Point it at the Homebrew OpenJDK if unset."""
    if os.environ.get("JAVA_HOME"):
        return
    for candidate in (
        "/opt/homebrew/opt/openjdk@17",
        "/opt/homebrew/opt/openjdk",
        "/usr/local/opt/openjdk@17",
    ):
        if Path(candidate).exists():
            os.environ["JAVA_HOME"] = candidate
            return


def get_spark(app_name: str = "spine", shuffle_partitions: int = 8):
    """Build a local Delta-Lake-enabled SparkSession.

    Uses delta-spark's configure_spark_with_delta_pip so the matching Delta
    package is pulled and the Delta SQL extensions are registered.
    """
    _ensure_java_home()
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.warehouse.dir", str(resolve("data/spark-warehouse")))
        # Quieter local runs.
        .config("spark.ui.showConsoleProgress", "false")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
