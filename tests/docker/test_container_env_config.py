"""
Container environment configuration validations for Polars and related settings.
"""

from __future__ import annotations

import pytest

from tests.docker.conftest import docker_exec_python


@pytest.mark.parametrize(
    "container,keys",
    [
        ("spark-master", ["POLARS_MAX_THREADS", "PYARROW_IGNORE_TIMEZONE", "TZ"]),
        ("spark-worker", ["POLARS_MAX_THREADS", "PYARROW_IGNORE_TIMEZONE", "TZ"]),
        ("jupyter", ["POLARS_MAX_THREADS"]),
        ("airflow-webserver", ["POLARS_MAX_THREADS"]),
        ("airflow-scheduler", ["POLARS_MAX_THREADS"]),
    ],
)
def test_expected_env_variables_present(container: str, keys, docker_containers, polars_env):
    if container not in docker_containers:
        pytest.skip(f"{container} not running")

    code = """
import os
print(";".join([f"{k}={os.getenv(k,'')}" for k in ['KEYS_PLACEHOLDER']]))
""".replace("KEYS_PLACEHOLDER", "','".join(keys))
    rc, out, err = docker_exec_python(container, code, env=polars_env)
    assert rc == 0, f"env check failed in {container}: {err or out}"
    pairs = [seg for seg in out.strip().split(";") if seg]
    for k in keys:
        found = [p for p in pairs if p.startswith(k+"=")]
        assert found and found[0].split("=",1)[1] != "", f"{k} not set in {container}"

