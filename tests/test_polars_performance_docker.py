#!/usr/bin/env python3
"""
Polars Performance Test Suite (Docker)

Benchmarks Polars vs Pandas and validates streaming, memory efficiency,
and ML integration performance by executing workloads inside Docker containers
(jupyter preferred). Skips gracefully if Docker/containers are unavailable.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from typing import Dict, Tuple

import pytest


def _run(cmd, timeout=180) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode(), p.stderr.decode()


def _compose_cmd() -> list:
    try:
        if subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0:
            return ["docker", "compose"]
    except FileNotFoundError:
        pass
    return ["docker-compose"]


def _container_running(name: str) -> bool:
    rc, out, _ = _run(_compose_cmd() + ["ps", "--format", "json"], timeout=30)
    if rc != 0 or not out.strip():
        return False
    try:
        entries = json.loads(out)
        for e in entries:
            if e.get("Name") == name or e.get("Service") == name:
                return True
    except Exception:
        return False
    return False


def _exec_python(container: str, code: str, env: Dict[str, str] | None = None, timeout: int = 300):
    env_parts = []
    if env:
        for k, v in env.items():
            env_parts.extend(["env", f"{k}={v}"])
    full = f"docker exec -i {shlex.quote(container)} {' '.join(env_parts)} python - <<'PY'\n{code}\nPY\n"
    return _run(["bash", "-lc", full], timeout=timeout)


@pytest.fixture(scope="module")
def jupyter_available():
    if not _container_running("jupyter"):
        pytest.skip("jupyter container not running; start docker stack first")
    return True


def test_polars_vs_pandas_feature_engineering(jupyter_available):
    code = """
import time, numpy as np
import pandas as pd
import polars as pl

def pandas_fe(df):
    df = df.copy()
    df['x1x2'] = df['x1'] * df['x2']
    df['xsum'] = df['x1'] + df['x2']
    return df

def polars_fe(df):
    return df.with_columns([
        (pl.col('x1') * pl.col('x2')).alias('x1x2'),
        (pl.col('x1') + pl.col('x2')).alias('xsum')
    ])

N = 400000
pd_df = pd.DataFrame({'x1': np.random.randn(N), 'x2': np.random.randn(N)})
pl_df = pl.from_pandas(pd_df)

t0 = time.time(); _ = pandas_fe(pd_df); tp = time.time() - t0
t1 = time.time(); _ = polars_fe(pl_df); tl = time.time() - t1
print(f"pandas_sec={tp:.4f}")
print(f"polars_sec={tl:.4f}")
"""
    rc, out, err = _exec_python("jupyter", code, env={"POLARS_MAX_THREADS": os.getenv("POLARS_MAX_THREADS", "4")}, timeout=240)
    assert rc == 0, f"exec failed: {err or out}"
    vals = {k: float(v) for k, v in (ln.split("=") for ln in out.strip().splitlines() if "_sec=" in ln)}
    assert vals["polars_sec"] <= vals["pandas_sec"] * 1.2, f"Polars should be comparable/faster. Got {vals}"


def test_streaming_performance(jupyter_available):
    code = """
import polars as pl, time, os, numpy as np
path = '/home/jovyan/data/stream_test.csv'
N = 600000
if not os.path.exists(path):
    import pandas as pd
    pd.DataFrame({'a': np.arange(N), 'g': (np.arange(N)%23)}).to_csv(path, index=False)

t0 = time.time()
df = pl.scan_csv(path).group_by('g').agg([pl.len().alias('cnt')]).collect(streaming=True)
t = time.time() - t0
print(f"stream_sec={t:.4f}")
print(f"rows={int(df.select(pl.col('cnt').sum()).item())}")
"""
    rc, out, err = _exec_python("jupyter", code, env={"POLARS_MAX_THREADS": os.getenv("POLARS_MAX_THREADS", "4")}, timeout=300)
    assert rc == 0, f"streaming failed: {err or out}"
    stats = {k: float(v) if k == "stream_sec" else int(v) for k, v in (ln.split("=") for ln in out.strip().splitlines())}
    assert stats["stream_sec"] < 4.5, f"Streaming too slow: {stats}"
    assert stats["rows"] == 600000, f"Row count mismatch: {stats}"


def test_memory_efficiency(jupyter_available):
    code = """
import polars as pl
def rss_kb():
    try:
        with open('/proc/self/status','r') as f:
            for ln in f:
                if ln.startswith('VmRSS:'): return int(ln.split()[1])
    except Exception: return -1
    return -1
before = rss_kb()
N = 500000
df = pl.DataFrame({'a': pl.arange(0,N), 'b': (pl.arange(0,N)%11)})
out = df.lazy().group_by('b').agg([pl.len().alias('cnt')]).collect(streaming=True)
after = rss_kb()
print(f"rss_before={before}")
print(f"rss_after={after}")
"""
    rc, out, err = _exec_python("jupyter", code, env={"POLARS_MAX_THREADS": os.getenv("POLARS_MAX_THREADS", "4")})
    assert rc == 0, f"mem check failed: {err or out}"
    vals = {k: int(v) for k, v in (ln.split("=") for ln in out.strip().splitlines())}
    # Heuristic: RSS shouldn't grow more than ~500MB
    assert vals["rss_after"] - vals["rss_before"] < 500_000, f"Excessive memory growth: {vals}"


def test_ml_integration_performance(jupyter_available):
    code = """
import polars as pl, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb, lightgbm as lgb

N = 30000
df = pl.DataFrame({
    'x1': np.random.randn(N),
    'x2': np.random.randn(N),
    'y': (np.random.rand(N) > 0.5).astype(int),
})
feat = df.with_columns([(pl.col('x1')*pl.col('x2')).alias('x1x2'), (pl.col('x1')+pl.col('x2')).alias('xsum')])
pd_df = feat.to_pandas()
X = pd_df[['x1','x2','x1x2','xsum']]
y = pd_df['y']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7)

t0 = time.time()
xgbm = xgb.XGBClassifier(n_estimators=60, max_depth=4, learning_rate=0.2, subsample=0.9, n_jobs=2, tree_method='hist').fit(Xtr, ytr)
xgb_t = time.time()-t0
xgb_a = accuracy_score(yte, xgbm.predict(Xte))

t0 = time.time()
lgbm = lgb.LGBMClassifier(n_estimators=80, max_depth=-1, learning_rate=0.2, subsample=0.9, n_jobs=2).fit(Xtr, ytr)
lgb_t = time.time()-t0
lgb_a = accuracy_score(yte, lgbm.predict(Xte))

print(f"xgb_sec={xgb_t:.3f}")
print(f"xgb_acc={xgb_a:.3f}")
print(f"lgb_sec={lgb_t:.3f}")
print(f"lgb_acc={lgb_a:.3f}")
"""
    rc, out, err = _exec_python("jupyter", code, env={"POLARS_MAX_THREADS": os.getenv("POLARS_MAX_THREADS", "4")}, timeout=360)
    assert rc == 0, f"ml perf failed: {err or out}"
    stats = {k: float(v) for k, v in (ln.split("=") for ln in out.strip().splitlines())}
    # Loose thresholds to accommodate container variance
    assert stats["xgb_sec"] < 12.0 and stats["lgb_sec"] < 12.0, f"Training too slow: {stats}"
    assert stats["xgb_acc"] >= 0.5 and stats["lgb_acc"] >= 0.5, f"Poor accuracy: {stats}"

