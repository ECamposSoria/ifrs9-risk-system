"""Tests for DataprocManager local fallback behavior when GCP is disabled."""

from __future__ import annotations

import importlib
import sys
from typing import Dict

import pytest

MODULE_NAME = "src.gcp_integrations"


@pytest.fixture
def gcp_disabled_module(monkeypatch, tmp_path):
    monkeypatch.setenv("GCP_ENABLED", "false")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path / "storage"))
    monkeypatch.setenv("LOCAL_DB_PATH", str(tmp_path / "local.db"))
    sys.modules.pop(MODULE_NAME, None)
    module = importlib.import_module(MODULE_NAME)
    return module


def test_dataproc_local_registry_tracks_clusters(gcp_disabled_module):
    module = gcp_disabled_module
    manager = module.DataprocManager(project_id="local-project")

    assert manager.gcp_enabled is False
    assert manager.provision_dataproc_cluster("cluster-one", worker_count=3) is True
    assert "cluster-one" in manager._local_clusters

    details: Dict[str, object] = manager._local_clusters["cluster-one"]
    assert details["worker_count"] == 3
    assert details["preemptible_workers"] == 0

    # Idempotent provisioning returns success without duplicating state
    assert manager.provision_dataproc_cluster("cluster-one") is True
    assert list(manager._local_clusters.keys()).count("cluster-one") == 1

    # Deleting releases the registry entry
    assert manager.delete_dataproc_cluster("cluster-one") is True
    assert "cluster-one" not in manager._local_clusters

    # Deleting again reports failure to mirror NotFound semantics
    assert manager.delete_dataproc_cluster("cluster-one") is False
