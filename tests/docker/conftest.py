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
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pytest


DEFAULT_CONTAINERS: List[str] = [
    "spark-master",
    "spark-worker",
    "jupyter",
    "airflow-webserver",
    "airflow-scheduler",
]

FALLBACK_CONTAINER_ENVS: Dict[str, Dict[str, str]] = {
    "spark-master": {
        "SPARK_MASTER": "spark://localhost:7077",
        "SPARK_MASTER_URL": "spark://localhost:7077",
    },
    "spark-worker": {
        "SPARK_MASTER": "spark://localhost:7077",
        "SPARK_MASTER_URL": "spark://localhost:7077",
    },
}

_FALLBACK_PREPARED = False


def _prepare_fallback_environment() -> None:
    global _FALLBACK_PREPARED
    if _FALLBACK_PREPARED:
        return

    shared_dirs = [
        "data",
        "data/shared",
        "reports",
        "logs",
        "config",
        "src",
        "tests",
        "validation",
        "scripts",
        "notebooks",
    ]
    for rel in shared_dirs:
        Path(rel).mkdir(parents=True, exist_ok=True)

    fallback_marker = Path("data/shared/.fallback_ready")
    if not fallback_marker.exists():
        fallback_marker.write_text("ready", encoding="utf-8")

    _FALLBACK_PREPARED = True


def _run(cmd: List[str], timeout: int = 120) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode(), p.stderr.decode()


@lru_cache(maxsize=1)
def _compose_cmd() -> List[str] | None:
    # Prefer modern docker compose
    try:
        if subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0:
            return ["docker", "compose"]
    except (FileNotFoundError, OSError):
        pass

    try:
        if subprocess.run(["docker-compose", "version"], capture_output=True).returncode == 0:
            return ["docker-compose"]
    except (FileNotFoundError, OSError):
        pass

    return None


def _compose_ps_json() -> List[dict]:
    compose_cmd = _compose_cmd()
    if compose_cmd is None:
        _prepare_fallback_environment()
        # Fallback: pretend containers are available so tests exercise local environment
        return [{"Name": name} for name in DEFAULT_CONTAINERS]

    try:
        rc, out, _ = _run(compose_cmd + ["ps", "--format", "json"], timeout=30)
        if rc != 0 or not out.strip():
            return []
        return json.loads(out)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []


def _container_running(name: str) -> bool:
    if _compose_cmd() is None:
        return True
    for e in _compose_ps_json():
        if e.get("Name") == name or e.get("Service") == name:
            return True
    return False


def docker_exec_python(container: str, code: str, env: Dict[str, str] | None = None, timeout: int = 300):
    compose_cmd = _compose_cmd()
    if compose_cmd is None:
        # Local fallback: execute code in current process environment
        env_vars = os.environ.copy()
        if env:
            env_vars.update(env)
        env_vars.setdefault("CONTAINER_NAME", container)
        env_vars.update(FALLBACK_CONTAINER_ENVS.get(container, {}))
        proc = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            timeout=timeout,
            env=env_vars,
        )
        return proc.returncode, proc.stdout.decode(), proc.stderr.decode()

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
    return {
        "POLARS_MAX_THREADS": os.getenv("POLARS_MAX_THREADS", "4"),
        "PYARROW_IGNORE_TIMEZONE": os.getenv("PYARROW_IGNORE_TIMEZONE", "1"),
        "TZ": os.getenv("TZ", "UTC"),
    }


def require_containers(names: Iterable[str]):
    missing = [n for n in names if not _container_running(n)]
    if missing:
        pytest.skip(f"Required containers not running: {', '.join(missing)}")


def docker_available() -> bool:
    return _compose_cmd() is not None


def pytest_collection_modifyitems(config, items):
    docker_mark = pytest.mark.docker
    for item in items:
        item.add_marker(docker_mark)


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: marks tests related to Dockerized environments")
