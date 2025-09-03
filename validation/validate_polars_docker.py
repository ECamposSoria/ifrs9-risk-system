#!/usr/bin/env python3
"""
Docker Container Polars Validation Script

Runs IFRS9-focused validation of Polars within all primary containers and
validates cross-container data sharing, basic performance, memory behavior,
and ML integration (XGBoost, LightGBM via Polars workflows).

Targets:
- spark-master
- spark-worker
- jupyter
- airflow-webserver (optionally airflow-scheduler)

Outputs concise PASS/FAIL with detailed JSON report.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --------------------------- Configuration ---------------------------------

DEFAULT_CONTAINERS = [
    "spark-master",
    "spark-worker",
    "jupyter",
    "airflow-webserver",
]

# Shared host folder mapped into containers (per docker-compose.ifrs9.yml)
HOST_SHARED_DATA = Path("./data").resolve()
HOST_REPORTS_DIR = Path("./reports").resolve()
REPORTS_SUBDIR = HOST_REPORTS_DIR / "polars_docker_validation"

# Map container -> data mount path (compose differs by service)
CONTAINER_DATA_PATHS = {
    "spark-master": "/data",
    "spark-worker": "/data",
    "jupyter": "/home/jovyan/data",
    "airflow-webserver": "/opt/airflow/data",
    "airflow-scheduler": "/opt/airflow/data",
}

# Docker compose command detection
def detect_compose_command() -> List[str]:
    for cmd in ("docker", "docker-compose"):
        try:
            if cmd == "docker":
                # Prefer 'docker compose' if available
                out = subprocess.run(["docker", "compose", "version"], capture_output=True)
                if out.returncode == 0:
                    return ["docker", "compose"]
            else:
                out = subprocess.run(["docker-compose", "version"], capture_output=True)
                if out.returncode == 0:
                    return ["docker-compose"]
        except FileNotFoundError:
            continue
    return ["docker", "compose"]  # best-effort default


COMPOSE = detect_compose_command()


def run(cmd: List[str], timeout: int = 120) -> Tuple[int, str, str]:
    """Run a subprocess, returning (rc, stdout, stderr)."""
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    return proc.returncode, out, err


def docker_exec(container: str, py_code: str, env: Optional[Dict[str, str]] = None, timeout: int = 300) -> Tuple[int, str, str]:
    """Execute inline Python code inside a running container.

    Uses: docker exec -i <container> env KEY=VAL ... python - <<'PY' ... PY
    """
    env_parts = []
    if env:
        for k, v in env.items():
            env_parts.extend(["env", f"{k}={v}"])
    # We use bash -lc to ensure heredoc is handled correctly
    code = f"python - <<'PY'\n{py_code}\nPY\n"
    cmd = ["bash", "-lc", f"docker exec -i {shlex.quote(container)} {' '.join(env_parts)} {code}"]
    return run(cmd, timeout=timeout)


def ensure_dirs():
    HOST_SHARED_DATA.mkdir(parents=True, exist_ok=True)
    REPORTS_SUBDIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ValidationResult:
    container: str
    polars_ok: bool
    version: Optional[str]
    cross_share_ok: bool
    perf_stats: Dict[str, float]
    memory_stats: Dict[str, int]
    ml_integration_ok: bool
    errors: List[str]


def test_polars_import(container: str) -> Tuple[bool, Optional[str], Optional[str]]:
    code = """
import polars as pl
print(pl.__version__)
"""
    rc, out, err = docker_exec(container, code, env={"POLARS_MAX_THREADS": os.getenv("POLARS_MAX_THREADS", "4")}, timeout=60)
    if rc == 0 and out.strip():
        return True, out.strip().splitlines()[-1].strip(), None
    return False, None, err.strip() or "polars import failed"


def cross_container_share(writer: str, readers: List[str]) -> Tuple[bool, Dict[str, str]]:
    """Write a Parquet with Polars from writer and read from readers."""
    fname = f"polars_xc_{int(time.time())}.parquet"
    host_path = HOST_SHARED_DATA / fname
    if host_path.exists():
        host_path.unlink(missing_ok=True)

    w_mount = CONTAINER_DATA_PATHS.get(writer)
    if not w_mount:
        return False, {"writer": f"No data mount mapping for {writer}"}

    # Write in writer
    write_code = f"""
import polars as pl, os
import random, time
df = pl.DataFrame({{
    'id': pl.arange(0, 1000),
    'bucket': (pl.arange(0,1000) % 10),
    'value': pl.arange(0, 1000) * 2
}})
df.write_parquet(os.path.join({w_mount!r}, {fname!r}))
print('OK')
"""
    rc_w, out_w, err_w = docker_exec(writer, write_code)
    notes: Dict[str, str] = {}
    if rc_w != 0:
        notes[writer] = f"write failed: {err_w.strip()}"
        return False, notes

    # Read in readers
    ok = True
    for r in readers:
        r_mount = CONTAINER_DATA_PATHS.get(r)
        if not r_mount:
            ok = False
            notes[r] = f"No data mount mapping for {r}"
            continue
        read_code = f"""
import polars as pl, os
import sys
df = pl.read_parquet(os.path.join({r_mount!r}, {fname!r}))
print(df.height, df['value'].sum())
"""
        rc_r, out_r, err_r = docker_exec(r, read_code)
        if rc_r != 0:
            ok = False
            notes[r] = f"read failed: {err_r.strip()}"
        else:
            last = out_r.strip().splitlines()[-1]
            notes[r] = last
            try:
                h, s = last.split()
                if int(h) != 1000 or int(s) != sum(i * 2 for i in range(1000)):
                    ok = False
                    notes[r] += " (mismatch)"
            except Exception:
                ok = False
                notes[r] += " (parse error)"

    # Cleanup from host side if present
    try:
        if host_path.exists():
            host_path.unlink()
    except Exception:
        pass

    return ok, notes


def basic_perf(container: str) -> Tuple[Dict[str, float], Optional[str]]:
    """Run a small Polars performance micro-benchmark inside a container."""
    code = """
import polars as pl, time
N = 300000
df = pl.DataFrame({{
    'id': pl.arange(0, N),
    'g': (pl.arange(0, N) % 50),
    'x': pl.arange(0, N).cast(pl.Float64)
}})
start = time.time()
res = df.group_by('g').agg([pl.col('x').mean().alias('mean_x'), pl.count().alias('cnt')])
t1 = time.time() - start
start = time.time()
joined = df.join(res, on='g', how='left')
t2 = time.time() - start
print(f"groupby_sec={t1:.4f}")
print(f"join_sec={t2:.4f}")
"""
    rc, out, err = docker_exec(container, code, env={"POLARS_MAX_THREADS": os.getenv("POLARS_MAX_THREADS", "4")}, timeout=180)
    stats: Dict[str, float] = {}
    if rc != 0:
        return stats, err.strip() or "perf failed"
    for line in out.strip().splitlines():
        if "groupby_sec=" in line or "join_sec=" in line:
            k, v = line.split("=", 1)
            try:
                stats[k] = float(v)
            except ValueError:
                pass
    return stats, None


def memory_test(container: str) -> Tuple[Dict[str, int], Optional[str]]:
    """Approximate memory usage before/after a streaming operation."""
    code = """
import polars as pl, os

def rss_kb():
    try:
        with open('/proc/self/status','r') as f:
            for ln in f:
                if ln.startswith('VmRSS:'):
                    return int(ln.split()[1])
    except Exception:
        return -1

before = rss_kb()
N = 400000
df = pl.DataFrame({{'a': pl.arange(0,N), 'b': (pl.arange(0,N) % 13)}})
scan = df.lazy()
out = scan.group_by('b').agg([pl.len().alias('cnt')]).collect(streaming=True)
after = rss_kb()
print(f"rss_kb_before={before}")
print(f"rss_kb_after={after}")
"""
    rc, out, err = docker_exec(container, code, env={"POLARS_MAX_THREADS": os.getenv("POLARS_MAX_THREADS", "4")}, timeout=180)
    stats: Dict[str, int] = {}
    if rc != 0:
        return stats, err.strip() or "memory test failed"
    for line in out.strip().splitlines():
        if line.startswith("rss_kb_"):
            k, v = line.split("=", 1)
            try:
                stats[k] = int(v)
            except ValueError:
                pass
    return stats, None


def ml_integration(container: str) -> Tuple[bool, Optional[str]]:
    """Train tiny models using Polars workflow to ensure integration."""
    code = """
import polars as pl
import numpy as np

N = 2000
df = pl.DataFrame({{
    'x1': np.random.randn(N),
    'x2': np.random.randn(N),
    'label': (np.random.rand(N) > 0.5).astype(int),
}})
# Simple feature engineering in Polars, then convert for ML libs
feat = df.with_columns([
    (pl.col('x1') * pl.col('x2')).alias('x1x2'),
    (pl.col('x1') + pl.col('x2')).alias('xsum')
])
pd_df = feat.to_pandas()
X = pd_df[['x1', 'x2', 'x1x2', 'xsum']]
y = pd_df['label']

ok = True
msgs = []
try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    model = xgb.XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.2, subsample=0.8, n_jobs=2, tree_method='hist')
    model.fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    print(f"xgb_acc={acc:.3f}")
except Exception as e:
    ok = False
    msgs.append(f"xgb:{e}")

try:
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    model = lgb.LGBMClassifier(n_estimators=40, max_depth=-1, learning_rate=0.2, subsample=0.9, n_jobs=2)
    model.fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    print(f"lgb_acc={acc:.3f}")
except Exception as e:
    ok = False
    msgs.append(f"lgb:{e}")

if not ok:
    import sys
    print(";".join(msgs))
    sys.exit(2)
"""
    rc, out, err = docker_exec(container, code, env={"POLARS_MAX_THREADS": os.getenv("POLARS_MAX_THREADS", "4")}, timeout=240)
    if rc == 0:
        return True, None
    return False, (err or out).strip()


def main() -> int:
    ensure_dirs()

    containers = DEFAULT_CONTAINERS.copy()
    # Allow CLI override
    if len(sys.argv) > 1:
        containers = sys.argv[1].split(",")

    # Validation run
    results: Dict[str, Dict] = {}
    overall_ok = True
    start_ts = datetime.now().isoformat()

    # Quick presence check for containers from compose ps
    rc, out, _ = run(COMPOSE + ["ps", "--format", "json"], timeout=60)
    running_names = set()
    if rc == 0 and out.strip():
        try:
            import json as _json
            for entry in _json.loads(out):
                name = entry.get("Name") or entry.get("Service")
                if name:
                    running_names.add(name)
        except Exception:
            pass

    for c in containers:
        entry = {
            "polars": {},
            "cross_share": {},
            "perf": {},
            "memory": {},
            "ml": {},
            "status": "PENDING",
        }
        errors: List[str] = []

        if running_names and c not in running_names:
            entry["status"] = "SKIPPED"
            entry["note"] = "container not running per compose ps"
            results[c] = entry
            overall_ok = False
            continue

        # Polars import/version
        ok_polars, version, err = test_polars_import(c)
        entry["polars"] = {"ok": ok_polars, "version": version, "error": err}
        if not ok_polars:
            errors.append(f"polars import failed: {err}")

        # Cross container share (skip writer is this container?)
        readers = [x for x in containers if x != c]
        ok_share, notes = cross_container_share(c, readers)
        entry["cross_share"] = {"ok": ok_share, "notes": notes}
        if not ok_share:
            errors.append("cross-container sharing failed")

        # Performance
        perf, errp = basic_perf(c)
        entry["perf"] = {"metrics": perf, "error": errp}
        # Soft thresholds: groupby under 2.5s, join under 2.5s for given N
        if perf:
            if perf.get("groupby_sec", 99.0) > 2.5 or perf.get("join_sec", 99.0) > 2.5:
                errors.append("performance thresholds not met")
        else:
            errors.append(f"performance run failed: {errp}")

        # Memory streaming
        mem, errm = memory_test(c)
        entry["memory"] = {"metrics": mem, "error": errm}
        if mem:
            before = mem.get("rss_kb_before", -1)
            after = mem.get("rss_kb_after", -1)
            if before > 0 and after > 0 and after - before > 400000:  # ~400MB
                errors.append("memory usage increased excessively")
        else:
            errors.append(f"memory test failed: {errm}")

        # ML Integration
        ok_ml, errmsg = ml_integration(c)
        entry["ml"] = {"ok": ok_ml, "error": errmsg}
        if not ok_ml:
            errors.append(f"ml integration failed: {errmsg}")

        entry["status"] = "PASS" if not errors else "FAIL"
        entry["errors"] = errors
        results[c] = entry
        overall_ok = overall_ok and (entry["status"] == "PASS")

    summary = {
        "started": start_ts,
        "finished": datetime.now().isoformat(),
        "overall_status": "success" if overall_ok else "failure",
        "containers": containers,
    }

    report = {"summary": summary, "results": results}
    out_path = REPORTS_SUBDIR / f"report_{int(time.time())}.json"
    try:
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved report: {out_path}")
    except Exception as e:
        print(f"WARN: failed to write report: {e}")

    # console summary
    print("POLARS DOCKER VALIDATION SUMMARY")
    for c, r in results.items():
        print(f"- {c}: {r.get('status')} | polars={r.get('polars',{}).get('version')} perf={r.get('perf',{}).get('metrics')}")

    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())

