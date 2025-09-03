"""
Pytest fixtures and Docker helpers for containerized Polars tests.

These fixtures execute small Python snippets inside running containers via
`docker exec`, validate availability of services, and provide container lists.

Marked tests use `@pytest.mark.docker` to allow filtering.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from typing import Dict, Iterable, List, Tuple

import pytest


DEFAULT_CONTAINERS: List[str] = [
    "spark-master",
    "spark-worker",
    "jupyter",
    "airflow-webserver",
    "airflow-scheduler",
]


def _run(cmd: List[str], timeout: int = 120) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode(), p.stderr.decode()


def _compose_cmd() -> List[str]:
    # Prefer modern docker compose
    try:
        if subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0:
            return ["docker", "compose"]
    except FileNotFoundError:
        pass
    return ["docker-compose"]


def _compose_ps_json() -> List[dict]:
    rc, out, _ = _run(_compose_cmd() + ["ps", "--format", "json"], timeout=30)
    if rc != 0 or not out.strip():
        return []
    try:
        return json.loads(out)
    except Exception:
        return []


def _container_running(name: str) -> bool:
    for e in _compose_ps_json():
        if e.get("Name") == name or e.get("Service") == name:
            return True
    return False


def docker_exec_python(container: str, code: str, env: Dict[str, str] | None = None, timeout: int = 300):
    env_parts: List[str] = []
    if env:
        for k, v in env.items():
            env_parts.extend(["env", f"{k}={v}"])
    snippet = f"docker exec -i {shlex.quote(container)} {' '.join(env_parts)} python - <<'PY'\n{code}\nPY\n"
    return _run(["bash", "-lc", snippet], timeout=timeout)


@pytest.fixture(scope="session")
def docker_containers() -> List[str]:
    # Allow override via ENV
    override = os.getenv("DOCKER_TEST_CONTAINERS")
    if override:
        wanted = [x.strip() for x in override.split(",") if x.strip()]
    else:
        wanted = DEFAULT_CONTAINERS
    # Filter by running
    running = [c for c in wanted if _container_running(c)]
    if not running:
        pytest.skip("No target docker containers are running for tests")
    return running


@pytest.fixture(scope="session")
def polars_env() -> Dict[str, str]:
    return {"POLARS_MAX_THREADS": os.getenv("POLARS_MAX_THREADS", "4")}


def require_containers(names: Iterable[str]):
    missing = [n for n in names if not _container_running(n)]
    if missing:
        pytest.skip(f"Required containers not running: {', '.join(missing)}")


pytestmark = pytest.mark.docker

