"""Additional parity checks for the Docker fallback harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.docker.conftest import docker_available, docker_exec_python


def test_shared_volume_roundtrip(polars_env):
    if docker_available():
        pytest.skip("Fallback parity checks only run without Docker")

    shared_file = Path("data/shared/fallback_roundtrip.txt")
    if shared_file.exists():
        shared_file.unlink()

    rc, out, err = docker_exec_python(
        "spark-master",
        """
from pathlib import Path
Path('data/shared').mkdir(parents=True, exist_ok=True)
Path('data/shared/fallback_roundtrip.txt').write_text('spark-master', encoding='utf-8')
""",
        env=polars_env,
    )
    assert rc == 0, f"spark-master write failed: {err or out}"

    rc, out, err = docker_exec_python(
        "spark-worker",
        """
from pathlib import Path
print(Path('data/shared/fallback_roundtrip.txt').read_text(encoding='utf-8'))
""",
        env=polars_env,
    )
    assert rc == 0, f"spark-worker read failed: {err or out}"
    assert out.strip() == "spark-master"


def test_fallback_injects_spark_environment(polars_env):
    if docker_available():
        pytest.skip("Fallback parity checks only run without Docker")

    rc, out, err = docker_exec_python(
        "spark-worker",
        """
import os
print(os.getenv('SPARK_MASTER_URL'))
""",
        env=polars_env,
    )
    assert rc == 0, f"env lookup failed: {err or out}"
    assert out.strip() == "spark://localhost:7077"
