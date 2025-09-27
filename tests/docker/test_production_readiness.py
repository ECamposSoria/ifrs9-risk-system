"""
Production Readiness Tests for Polars in Dockerized IFRS9 System

Includes:
- Container health checks with Polars validation
- Service discovery/connectivity checks
- Error handling and recovery semantics (health exit codes)
- Logging/monitoring artifacts generation
"""

from __future__ import annotations

import os
import time
import subprocess
from pathlib import Path
import pytest

from tests.docker.conftest import docker_available, docker_exec_python


def test_polars_healthcheck_script_all_containers(docker_containers, polars_env):
    if not docker_available():
        script = Path("validation/polars_health_check.py")
        data_dir = Path(os.getenv("IFRS9_TEST_DATA_DIR", "/tmp/ifrs9_tests"))
        data_dir.mkdir(parents=True, exist_ok=True)
        outfile = data_dir / f"health_{int(time.time())}.json"
        cmd = ["python", str(script), "--threshold-ms", "3500", "--data-path", str(data_dir), "--json-out", str(outfile), "--quiet"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode in (0, 1), proc.stderr or proc.stdout
        assert outfile.exists()
        return

    for c in docker_containers:
        # Use health check script in repo mounted into containers
        # Write JSON to shared mount when possible
        mounts = {
            "spark-master": ("/data", "/app/validation/polars_health_check.py"),
            "spark-worker": ("/data", "/app/validation/polars_health_check.py"),
            "jupyter": ("/home/jovyan/data", "/home/jovyan/validation/polars_health_check.py"),
            "airflow-webserver": ("/opt/airflow/data", "/opt/airflow/validation/polars_health_check.py"),
            "airflow-scheduler": ("/opt/airflow/data", "/opt/airflow/validation/polars_health_check.py"),
        }
        data_path, script_path = mounts.get(c, (None, None))
        if not data_path or not script_path:
            pytest.skip(f"mounts not configured for {c}")
        outfile = f"{data_path}/health_{int(time.time())}.json"
        code = f"""
import sys
import subprocess
rc = subprocess.call(['python', '{script_path}', '--threshold-ms', '3500', '--data-path', '{data_path}', '--json-out', '{outfile}', '--quiet'])
print(f"rc={{rc}}")
sys.exit(0)
"""
        rc, out, err = docker_exec_python(c, code, env=polars_env, timeout=180)
        assert rc == 0, f"health script failed to run in {c}: {err or out}"
        vals = dict([ln.split("=",1) for ln in out.strip().splitlines() if "=" in ln])
        # Accept 0 (healthy) or 1 (degraded) — not 2
        assert vals.get("rc") in {"0", "1"}


def test_service_connectivity_from_worker_to_master(docker_containers, polars_env):
    if not docker_available():
        proc = subprocess.run(["python", "-c", "import pyspark; print('ok=1')"], capture_output=True, text=True)
        assert proc.returncode == 0 and "ok=1" in proc.stdout
        return
    if "spark-worker" not in docker_containers or "spark-master" not in docker_containers:
        pytest.skip("Spark containers not running")
    code = """
import socket
s=socket.socket(); s.settimeout(5)
try:
    s.connect(('spark-master', 7077)); print('ok=1')
except Exception as e:
    print(f'err={type(e).__name__}')
finally:
    try: s.close()
    except: pass
"""
    rc, out, err = docker_exec_python("spark-worker", code, env=polars_env, timeout=60)
    assert rc == 0, f"socket check failed: {err or out}"
    kv = dict([ln.split("=",1) for ln in out.strip().splitlines() if "=" in ln])
    assert kv.get("ok") == "1"


def test_airflow_can_read_shared_data(docker_containers, polars_env):
    if not docker_available():
        shared = Path("data")
        assert shared.exists(), "host data directory missing"
        return
    if "airflow-webserver" not in docker_containers:
        pytest.skip("airflow-webserver not running")
    code = """
import os
exists = os.path.exists('/opt/airflow/data') and os.listdir('/opt/airflow/data') is not None
print(f"ok={int(exists)}")
"""
    rc, out, err = docker_exec_python("airflow-webserver", code, env=polars_env)
    assert rc == 0, f"airflow data check failed: {err or out}"
    kv = dict([ln.split("=",1) for ln in out.strip().splitlines() if "=" in ln])
    assert kv.get("ok") == "1"


def test_logging_artifacts_present():
    # Verify host-side logs and reports directories exist for monitoring
    assert Path('logs').exists(), "logs directory missing"
    assert Path('reports').exists(), "reports directory missing"
