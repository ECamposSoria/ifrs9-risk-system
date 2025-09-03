#!/usr/bin/env python3
"""
IFRS9 Agents Readiness Checker

Reads config/orchestration_rules.yaml and monitoring/prometheus.yml to derive
agent service names and ports, probes health endpoints, and emits a readiness report.
"""

from __future__ import annotations

import os
import sys
import json
import time
import socket
import logging
from typing import Dict, List, Tuple
from pathlib import Path

import yaml
import http.client

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _http_get(host: str, port: int, path: str = "/health", timeout: float = 3.0) -> Tuple[int, str]:
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="ignore")
        conn.close()
        return resp.status, body
    except Exception as e:
        return 0, str(e)


def load_targets() -> Dict[str, Tuple[str, int]]:
    targets: Dict[str, Tuple[str, int]] = {}
    prom = Path("monitoring/prometheus.yml")
    if prom.exists():
        cfg = yaml.safe_load(prom.read_text())
        for job in cfg.get("scrape_configs", []):
            name = job.get("job_name")
            if name and name.startswith("ifrs9-"):
                scs = job.get("static_configs", [])
                for sc in scs:
                    for t in sc.get("targets", []):
                        try:
                            host, port = t.split(":", 1)
                            # Assume default health endpoint on same host:port
                            targets[name] = (host, int(port))
                        except Exception:
                            continue
    # Fallback defaults if empty
    if not targets:
        defaults = {
            "ifrs9-rules-engine": ("localhost", 8001),
            "ifrs9-data-generator": ("localhost", 8002),
            "ifrs9-ml-models": ("localhost", 8003),
            "ifrs9-validator": ("localhost", 8004),
            "ifrs9-integrator": ("localhost", 8005),
            "ifrs9-orchestrator": ("localhost", 8006),
            "ifrs9-reporter": ("localhost", 8007),
            "ifrs9-debugger": ("localhost", 8008),
        }
        targets.update(defaults)
    return targets


def check_readiness() -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    targets = load_targets()
    for agent, (host, port) in targets.items():
        status, body = _http_get(host, port, "/health")
        results[agent] = {
            "host": host,
            "port": str(port),
            "status_code": str(status),
            "healthy": "true" if status == 200 else "false",
        }
        if status != 200:
            # try readiness
            status2, _ = _http_get(host, port, "/ready")
            if status2 == 200:
                results[agent]["healthy"] = "true"
                results[agent]["status_code"] = str(status2)
    return results


def main():
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agents": check_readiness(),
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    out = Path("reports/agents_readiness_report.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

