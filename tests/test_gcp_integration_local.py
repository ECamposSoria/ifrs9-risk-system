"""Tests for GCP integration fallback behavior when cloud features are disabled."""

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

MODULE_NAME = "src.gcp_integrations"


@pytest.fixture
def local_gcp_module(monkeypatch, tmp_path):
    """Reload the GCP integration module with local fallback configuration."""
    storage_dir = tmp_path / "storage"
    monkeypatch.setenv("GCP_ENABLED", "false")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(storage_dir))
    monkeypatch.setenv("LOCAL_DB_PATH", str(tmp_path / "local.db"))

    # ensure fresh module import to respect environment overrides
    sys.modules.pop(MODULE_NAME, None)
    module = importlib.import_module(MODULE_NAME)

    # force deterministic environment detection for assertions
    monkeypatch.setattr(module.GCPIntegration, "_detect_environment", lambda self: True, raising=False)
    return module


def test_local_storage_upload_persists_parquet(local_gcp_module, tmp_path):
    integration = local_gcp_module.GCPIntegration()
    assert integration.gcp_enabled is False

    df = pd.DataFrame({"loan_id": ["L001"], "amount": [1000.0]})
    result = integration.upload_dataframe_to_storage(df, "loan_snapshot")

    expected_file = Path(integration.local_storage_path) / "loan_snapshot.parquet"
    assert result is True
    assert expected_file.exists()


def test_environment_info_reports_local_paths(local_gcp_module, tmp_path):
    integration = local_gcp_module.GCPIntegration()
    info = integration.get_environment_info()

    assert info["gcp_enabled"] is False
    assert info["local_storage_path"] == str(integration.local_storage_path)
    assert info["local_db_path"] == integration.local_db_path
    assert info["is_local"] is True
