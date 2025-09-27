"""
Docker-Specific Polars Performance Tests

Benchmarks Polars vs Pandas, memory usage, streaming performance,
and cross-container data sharing using shared volumes.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict

import pytest

from tests.docker.conftest import docker_available, docker_exec_python


def test_cross_container_parquet_sharing(docker_containers, polars_env):
    # Write from first container, read in others
    fname = f"xc_{int(time.time())}.parquet"

    if not docker_available():
        base_dir = Path(os.getenv("IFRS9_TEST_DATA_DIR", "/tmp/ifrs9_tests"))
        base_dir.mkdir(parents=True, exist_ok=True)
        file_path = base_dir / fname

        import polars as pl

        idx = pl.arange(0, 1000, eager=True)
        df = pl.DataFrame({"id": idx, "v": idx * 2})
        df.write_parquet(file_path)

        loaded = pl.read_parquet(file_path)
        assert loaded.height == 1000
        assert int(loaded["v"].sum()) == sum(i * 2 for i in range(1000))
        return

    writer = docker_containers[0]
    mounts = {
        "spark-master": "/data",
        "spark-worker": "/data",
        "jupyter": "/home/jovyan/data",
        "airflow-webserver": "/opt/airflow/data",
        "airflow-scheduler": "/opt/airflow/data",
    }
    write_path = mounts.get(writer, "/data")
    rc, out, err = docker_exec_python(
        writer,
f"""
import polars as pl, os
idx = pl.arange(0, 1000, eager=True)
df = pl.DataFrame({{'id': idx, 'v': idx*2}})
df.write_parquet(os.path.join('{write_path}', '{fname}'))
print('ok=1')
""",
        env=polars_env,
    )
    assert rc == 0 and "ok=1" in out

    for reader in docker_containers[1:]:
        rpath = mounts.get(reader, "/data")
        rc, out, err = docker_exec_python(
            reader,
            f"""
import polars as pl, os
df = pl.read_parquet(os.path.join('{rpath}', '{fname}'))
print(f"rows={df.height}")
print(f"sum={int(df['v'].sum())}")
""",
            env=polars_env,
        )
        assert rc == 0, f"read failed in {reader}: {err or out}"
        stats = dict([ln.split("=",1) for ln in out.strip().splitlines() if "=" in ln])
        assert int(stats["rows"]) == 1000
        assert int(stats["sum"]) == sum(i*2 for i in range(1000))


@pytest.mark.parametrize("container", ["jupyter", "spark-master"])
def test_polars_vs_pandas_feature_engineering(container: str, docker_containers, polars_env: Dict[str, str]):
    if container not in docker_containers:
        pytest.skip(f"{container} not running")
    code = """
import time, numpy as np
import pandas as pd
import polars as pl
N = 250000
pd_df = pd.DataFrame({'x1': np.random.randn(N), 'x2': np.random.randn(N)})
pl_df = pl.from_pandas(pd_df)
t0=time.time(); _=pd_df.assign(x1x2=pd_df['x1']*pd_df['x2'], xsum=pd_df['x1']+pd_df['x2']); tp=time.time()-t0
t1=time.time(); _=pl_df.with_columns([(pl.col('x1')*pl.col('x2')).alias('x1x2'), (pl.col('x1')+pl.col('x2')).alias('xsum')]); tl=time.time()-t1
print(f"pandas_sec={tp:.4f}")
print(f"polars_sec={tl:.4f}")
"""
    rc, out, err = docker_exec_python(container, code, env=polars_env, timeout=240)
    assert rc == 0, f"perf exec failed in {container}: {err or out}"
    nums = {k: float(v) for k, v in (ln.split("=") for ln in out.strip().splitlines())}
    assert nums["polars_sec"] <= nums["pandas_sec"] * 1.25


@pytest.mark.parametrize("container", ["jupyter", "spark-master"])
def test_memory_usage_streaming(container: str, docker_containers, polars_env):
    if container not in docker_containers:
        pytest.skip(f"{container} not running")
    code = """
import polars as pl
def rss_kb():
    try:
        with open('/proc/self/status','r') as f:
            for ln in f:
                if ln.startswith('VmRSS:'): return int(ln.split()[1])
    except Exception: return -1
    return -1
before=rss_kb()
N=400000
idx = pl.arange(0, N, eager=True)
df = pl.DataFrame({'a': idx, 'g': (idx%17)})
_ = df.lazy().group_by('g').agg(pl.len().alias('cnt')).collect(streaming=True)
after=rss_kb()
print(f"before={before}")
print(f"after={after}")
"""
    rc, out, err = docker_exec_python(container, code, env=polars_env, timeout=240)
    assert rc == 0, f"mem streaming failed in {container}: {err or out}"
    vals = {k: int(v) for k, v in (ln.split("=") for ln in out.strip().splitlines())}
    assert vals["after"] - vals["before"] < 500_000
