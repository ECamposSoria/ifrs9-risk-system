"""GCP-enabled integration tests using stubbed Google Cloud clients."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Dict, List, Any, Optional

import pandas as pd
import pytest


def _install_stub_google_modules(monkeypatch, context: Dict) -> None:
    """Register stub google modules so src.gcp_integrations imports succeed."""

    # Exceptions module -------------------------------------------------
    exceptions_mod = ModuleType("google.cloud.exceptions")

    class NotFound(Exception):
        pass

    class Conflict(Exception):
        pass

    exceptions_mod.NotFound = NotFound
    exceptions_mod.Conflict = Conflict

    # BigQuery module ---------------------------------------------------
    bigquery_mod = ModuleType("google.cloud.bigquery")

    @dataclass
    class SchemaField:
        name: str
        field_type: str
        mode: str = "NULLABLE"

    class Dataset:
        def __init__(self, dataset_id: str):
            self.dataset_id = dataset_id
            self.location = None
            self.description = ""

    class Table:
        def __init__(self, table_id: str, schema: List[SchemaField]):
            self.table_id = table_id
            self.schema = schema
            self.description = ""

    class LoadJobConfig:
        def __init__(self, write_disposition: str = "WRITE_TRUNCATE", autodetect: bool = False):
            self.write_disposition = write_disposition
            self.autodetect = autodetect

    class _QueryJob:
        def __init__(self, query_text: str, metadata: Dict[str, Any]):
            self._query = query_text
            self._metadata = metadata

        def to_dataframe(self):
            context.setdefault("query_invocations", []).append(
                {**self._metadata, "query": self._query}
            )
            context.setdefault("queries", []).append(self._query)
            return context["query_result"].copy()

    class _LoadJob:
        def __init__(self, table_ref: str, rows: int, disposition: str | None, job_id: Optional[str]):
            self.table_ref = table_ref
            self.rows = rows
            self.disposition = disposition
            self.job_id = job_id

        def result(self):  # matches bigquery job API
            return None

    class Client:
        def __init__(self, project: str, credentials):
            self.project = project
            self.credentials = credentials

        def load_table_from_dataframe(
            self,
            df: pd.DataFrame,
            table_ref: str,
            job_config: LoadJobConfig | None = None,
            job_id: str | None = None,
            retry=None,
            **_: Dict[str, Any],
        ):
            load_errors = context.setdefault("load_errors", [])
            if load_errors:
                exc = load_errors.pop(0)
                raise exc
            if job_id:
                seen_ids = context.setdefault("load_job_ids", set())
                if job_id in seen_ids:
                    raise Conflict("duplicate job id")
                seen_ids.add(job_id)
            context["loads"].append(
                {
                    "table": table_ref,
                    "rows": len(df),
                    "disposition": job_config.write_disposition if job_config else None,
                    "job_id": job_id,
                    "retry": retry,
                }
            )
            disposition = job_config.write_disposition if job_config else None
            return _LoadJob(table_ref, len(df), disposition, job_id)

        def create_dataset(self, dataset: Dataset, timeout: int = 30):
            if dataset.dataset_id in context["datasets"]:
                raise Conflict("Dataset already exists")
            context["datasets"].add(dataset.dataset_id)
            return dataset

        def get_dataset(self, dataset_ref: str):
            if dataset_ref not in context["datasets"]:
                raise NotFound()
            return Dataset(dataset_ref)

        def create_table(self, table: Table):
            if table.table_id in context["tables"]:
                raise Conflict("Table already exists")
            context["tables"].add(table.table_id)
            return table

        def get_table(self, table_ref: str):
            if table_ref not in context["tables"]:
                raise NotFound()
            return Table(table_ref, [])

        def query(
            self,
            query_text: str,
            job_config=None,
            job_id: str | None = None,
            retry=None,
            request_id: str | None = None,
            api_method: str | None = None,
            **_: Dict[str, Any],
        ):
            query_errors = context.setdefault("query_errors", [])
            if query_errors:
                exc = query_errors.pop(0)
                raise exc
            metadata = {
                "job_id": job_id,
                "retry": retry,
                "request_id": request_id,
                "api_method": api_method,
                "job_config": job_config,
            }
            return _QueryJob(query_text, metadata)

    bigquery_mod.Client = Client
    bigquery_mod.Dataset = Dataset
    bigquery_mod.SchemaField = SchemaField
    bigquery_mod.Table = Table
    bigquery_mod.LoadJobConfig = LoadJobConfig

    # Storage module ----------------------------------------------------
    storage_mod = ModuleType("google.cloud.storage")

    class _Bucket:
        def __init__(self, name: str):
            self.name = name
            self.storage_class = "STANDARD"
            self.location = None

    class StorageClient:
        def __init__(self, project: str, credentials):
            self.project = project
            self.credentials = credentials

        def get_bucket(self, name: str):
            if name not in context["buckets"]:
                raise NotFound()
            return context["buckets"][name]

        def bucket(self, name: str):
            return _Bucket(name)

        def create_bucket(self, bucket: _Bucket):
            if bucket.name in context["buckets"]:
                raise Conflict("Bucket exists")
            context["buckets"][bucket.name] = bucket
            return bucket

    storage_mod.Client = StorageClient

    # Dataproc / retry placeholders ------------------------------------
    dataproc_mod = ModuleType("google.cloud.dataproc_v1")
    retry_mod = ModuleType("google.api_core.retry")

    # service_account module -------------------------------------------
    service_account_mod = ModuleType("google.oauth2.service_account")

    class Credentials:
        def __init__(self, source: str | None):
            self.source = source

        @classmethod
        def from_service_account_file(cls, path: str):
            return cls(path)

    service_account_mod.Credentials = Credentials

    oauth2_mod = ModuleType("google.oauth2")
    oauth2_mod.service_account = service_account_mod

    # google root package ----------------------------------------------
    google_mod = ModuleType("google")
    cloud_mod = ModuleType("google.cloud")
    cloud_mod.bigquery = bigquery_mod
    cloud_mod.storage = storage_mod
    cloud_mod.dataproc_v1 = dataproc_mod
    cloud_mod.exceptions = exceptions_mod

    google_mod.cloud = cloud_mod
    google_mod.api_core = ModuleType("google.api_core")
    google_mod.api_core.retry = retry_mod
    google_mod.oauth2 = oauth2_mod

    modules = {
        "google": google_mod,
        "google.cloud": cloud_mod,
        "google.cloud.bigquery": bigquery_mod,
        "google.cloud.storage": storage_mod,
        "google.cloud.exceptions": exceptions_mod,
        "google.cloud.dataproc_v1": dataproc_mod,
        "google.oauth2": oauth2_mod,
        "google.oauth2.service_account": service_account_mod,
        "google.api_core": google_mod.api_core,
        "google.api_core.retry": retry_mod,
    }

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def gcp_enabled_module(monkeypatch, tmp_path):
    monkeypatch.setenv("GCP_ENABLED", "true")
    # Supply credentials file path
    cred_path = tmp_path / "service-account.json"
    cred_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred_path))

    context = {
        "datasets": set(),
        "tables": set(),
        "loads": [],
        "queries": [],
        "query_result": pd.DataFrame({"loan_id": ["L1"], "loan_amount": [1000.0]}),
        "buckets": {},
        "credentials_path": cred_path,
    }

    _install_stub_google_modules(monkeypatch, context)

    sys.modules.pop("src.gcp_integrations", None)
    module = importlib.import_module("src.gcp_integrations")
    return module, context


def test_bigquery_operations(gcp_enabled_module):
    module, ctx = gcp_enabled_module
    manager = module.BigQueryManager(project_id="demo-project", credentials_path=str(ctx["credentials_path"]))

    assert manager.gcp_enabled is True

    assert manager.create_dataset("ifrs9_data") is True
    assert "demo-project.ifrs9_data" in ctx["datasets"]

    # Second invocation should detect existing dataset
    assert manager.create_dataset("ifrs9_data") is True

    schema = [module.bigquery.SchemaField("loan_id", "STRING", mode="REQUIRED")]
    assert manager.create_table_from_schema("ifrs9_data", "loan_snapshot", schema) is True
    assert "demo-project.ifrs9_data.loan_snapshot" in ctx["tables"]

    df = pd.DataFrame({"loan_id": ["L1"], "loan_amount": [1000.0]})
    assert manager.upload_dataframe(df, "ifrs9_data", "loan_snapshot") is True
    assert ctx["loads"] and ctx["loads"][0]["table"] == "demo-project.ifrs9_data.loan_snapshot"

    ctx["query_result"] = pd.DataFrame({"loan_id": ["L1"], "pd": [0.02]})
    result = manager.query_to_dataframe("SELECT loan_id, pd FROM demo-project.ifrs9_data.loan_snapshot")
    assert list(result.columns) == ["loan_id", "pd"]
    assert not result.empty


def test_bigquery_idempotent_load_and_retry(gcp_enabled_module):
    module, ctx = gcp_enabled_module
    manager = module.BigQueryManager(project_id="demo-project", credentials_path=str(ctx["credentials_path"]))

    manager.create_dataset("ifrs9_data")
    df = pd.DataFrame({"loan_id": ["L1"], "loan_amount": [1000.0]})
    job_id = "ifrs9-load-001"
    retry_sentinel = object()

    assert manager.upload_dataframe(
        df,
        "ifrs9_data",
        "snapshots",
        job_id=job_id,
        retry_config=retry_sentinel,
    ) is True
    assert ctx["loads"][-1]["job_id"] == job_id
    assert ctx["loads"][-1]["retry"] is retry_sentinel

    before = len(ctx["loads"])
    assert manager.upload_dataframe(
        df,
        "ifrs9_data",
        "snapshots",
        job_id=job_id,
        retry_config=retry_sentinel,
    ) is True
    assert len(ctx["loads"]) == before  # Conflict treated as idempotent success
    assert job_id in ctx["load_job_ids"]


def test_bigquery_query_request_id_tracks_retry(gcp_enabled_module):
    module, ctx = gcp_enabled_module
    manager = module.BigQueryManager(project_id="demo-project", credentials_path=str(ctx["credentials_path"]))

    request_id = "abc-123"
    retry_sentinel = object()
    result = manager.query_to_dataframe(
        "SELECT 1",
        request_id=request_id,
        retry_config=retry_sentinel,
    )
    assert result is not None
    invocation = ctx["query_invocations"][-1]
    assert invocation["request_id"] == request_id
    assert invocation["retry"] is retry_sentinel


def test_cloud_storage_bucket_management(gcp_enabled_module):
    module, ctx = gcp_enabled_module
    storage_manager = module.CloudStorageManager(project_id="demo-project", credentials_path=str(ctx["credentials_path"]))

    assert storage_manager.create_bucket("ifrs9-artifacts") is True
    assert "ifrs9-artifacts" in ctx["buckets"]
    # Existing bucket should be treated as success
    assert storage_manager.create_bucket("ifrs9-artifacts") is True


def test_bigquery_error_paths(gcp_enabled_module):
    module, ctx = gcp_enabled_module
    manager = module.BigQueryManager(project_id="demo-project", credentials_path=str(ctx["credentials_path"]))

    # First creation succeeds, second surfaces Conflict but should be handled gracefully
    assert manager.create_dataset("ifrs9_events") is True
    assert manager.create_dataset("ifrs9_events") is True

    schema = [module.bigquery.SchemaField("loan_id", "STRING", mode="REQUIRED")]
    assert manager.create_table_from_schema("ifrs9_events", "events", schema) is True
    assert manager.create_table_from_schema("ifrs9_events", "events", schema) is True

    # Upload failure propagates as False and leaves load_errors empty afterwards
    ctx.setdefault("load_errors", []).append(RuntimeError("simulated failure"))
    df = pd.DataFrame({"loan_id": ["L1"], "loan_amount": [1000.0]})
    assert manager.upload_dataframe(df, "ifrs9_events", "events") is False
    assert ctx["load_errors"] == []

    # Query failure returns None and consumes queued exception
    ctx.setdefault("query_errors", []).append(module.NotFound("missing job"))
    assert manager.query_to_dataframe("SELECT 1") is None
