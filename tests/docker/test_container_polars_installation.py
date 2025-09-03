"""
Docker Container Polars Installation Tests

Validates Polars installation, version compatibility, feature flags, and
environment configuration across core containers.
"""

from __future__ import annotations

import os
from typing import Dict

import pytest

from .conftest import docker_exec_python


def _polars_info_code() -> str:
    return """
import os
import sys
try:
    import polars as pl
    print(f"ok=1")
    print(f"version={pl.__version__}")
    # Basic feature/config checks
    try:
        import pyarrow as pa
        print("pyarrow=1")
    except Exception:
        print("pyarrow=0")
    # Config sanity
    print(f"env_POLARS_MAX_THREADS={os.getenv('POLARS_MAX_THREADS','')}")
    # Simple op
    df = pl.DataFrame({'a':[1,2,3],'b':[2,3,4]})
    _ = df.with_columns((pl.col('a')+pl.col('b')).alias('c')).height
    print("df_ok=1")
except Exception as e:
    print("ok=0")
    print(f"error={type(e).__name__}:{e}")
    sys.exit(2)
"""


@pytest.mark.parametrize("container", [
    "spark-master", "spark-worker", "jupyter", "airflow-webserver", "airflow-scheduler"
])
def test_polars_installed_and_configured(container: str, docker_containers, polars_env: Dict[str, str]):
    if container not in docker_containers:
        pytest.skip(f"{container} not running")
    rc, out, err = docker_exec_python(container, _polars_info_code(), env=polars_env, timeout=90)
    assert rc == 0, f"polars check failed in {container}: {err or out}"
    lines = dict([ln.split("=", 1) for ln in out.strip().splitlines() if "=" in ln])
    assert lines.get("ok") == "1"
    assert lines.get("version") is not None and len(lines["version"]) > 0
    assert lines.get("df_ok") == "1"
    # Environment propagated
    assert lines.get("env_POLARS_MAX_THREADS") in {"2", "3", "4", "6", "8", os.getenv("POLARS_MAX_THREADS", "4")}


def test_polars_feature_flags(docker_containers, polars_env):
    # Verify streaming collect and lazy are available
    code = """
import polars as pl
df = pl.DataFrame({'g': (pl.arange(0, 10000) % 13), 'x': pl.arange(0,10000)})
out = df.lazy().group_by('g').agg(pl.len().alias('cnt')).collect(streaming=True)
print(f"ok={int(out.height>0)}")
"""
    for c in docker_containers:
        rc, out, err = docker_exec_python(c, code, env=polars_env)
        assert rc == 0, f"streaming/lazy failed in {c}: {err or out}"
        kv = dict([ln.split("=", 1) for ln in out.strip().splitlines() if "=" in ln])
        assert kv.get("ok") == "1"

