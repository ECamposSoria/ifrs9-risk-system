#!/usr/bin/env python3
"""Cloud Run batch job entrypoint for generating IFRS9 staging outputs."""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Ensure local package imports work when the module is executed from /app
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import importlib.util

def _load_data_generator():
    module_path = SRC_DIR / "generate_data.py"
    spec = importlib.util.spec_from_file_location("generate_data", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - safety
        raise ImportError(f"Unable to load generate_data.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DataGenerator  # type: ignore[attr-defined]

DataGenerator = _load_data_generator()

try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover - optional dependency handling
    storage = None

try:
    from google.cloud import bigquery  # type: ignore
except Exception:  # pragma: no cover - optional dependency handling
    bigquery = None


@dataclass
class StageSummary:
    provision_stage: str
    loan_count: int
    total_exposure: float


_SHUTDOWN_REQUESTED = False


def _handle_signal(signum, _frame):
    global _SHUTDOWN_REQUESTED
    logging.warning("Received signal %s – finishing current work and shutting down", signum)
    _SHUTDOWN_REQUESTED = True


def configure_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def generate_portfolio(record_count: int) -> pd.DataFrame:
    generator = DataGenerator(seed=int(os.getenv("IFRS9_SEED", "42")))
    logging.info("Generating synthetic portfolio with %s records", record_count)
    df = generator.generate_loan_portfolio(n_loans=record_count)

    if "created_date" not in df.columns:
        df["created_date"] = pd.to_datetime(df["created_at"], errors="coerce").dt.date
    return df


def summarise_portfolio(df: pd.DataFrame) -> List[StageSummary]:
    summary = (
        df.groupby("provision_stage")
        .agg(loan_count=("loan_id", "count"), total_exposure=("loan_amount", "sum"))
        .reset_index()
    )
    results: List[StageSummary] = []
    for row in summary.itertuples(index=False):
        results.append(StageSummary(row.provision_stage, int(row.loan_count), float(row.total_exposure)))
    logging.info("Stage summary computed for %s stages", len(results))
    return results


def write_local_outputs(df: pd.DataFrame, summary: List[StageSummary], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_path = output_dir / "loan_portfolio.parquet"
    summary_path = output_dir / "stage_summary.json"
    manifest_path = output_dir / "run_manifest.json"

    logging.info("Writing portfolio to %s", portfolio_path)
    df.to_parquet(portfolio_path, index=False)

    summary_payload = [asdict(item) for item in summary]
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": int(df.shape[0]),
        "stage_summary": summary_payload,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "portfolio": str(portfolio_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
    }


def maybe_upload_to_gcs(paths: dict, bucket_name: Optional[str], object_prefix: str) -> None:
    if not bucket_name:
        logging.info("No GCS bucket configured; skipping upload.")
        return
    if storage is None:
        logging.warning("google-cloud-storage not available; skipping upload to gs://%s", bucket_name)
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for label, path in paths.items():
        blob_name = f"{object_prefix.rstrip('/')}/{timestamp}/{Path(path).name}"
        logging.info("Uploading %s to gs://%s/%s", path, bucket_name, blob_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(path)

def _resolve_bigquery_write_disposition(value: str) -> str:
    if bigquery is None:
        return value
    candidate = value.strip().upper()
    return getattr(bigquery.WriteDisposition, candidate, bigquery.WriteDisposition.WRITE_TRUNCATE)


def maybe_write_to_bigquery(
    df: pd.DataFrame,
    project_id: Optional[str],
    dataset_id: Optional[str],
    table_id: str,
    *,
    location: Optional[str] = None,
    write_disposition: str = "WRITE_TRUNCATE",
) -> None:
    if not dataset_id:
        logging.info("No BigQuery dataset configured; skipping BigQuery load.")
        return
    if bigquery is None:
        logging.warning("google-cloud-bigquery not available; skipping load to dataset %s", dataset_id)
        return

    client = bigquery.Client(project=project_id) if project_id else bigquery.Client()
    resolved_project = project_id or client.project
    table_ref = f"{resolved_project}.{dataset_id}.{table_id}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=_resolve_bigquery_write_disposition(write_disposition),
    )

    logging.info("Loading %d rows to BigQuery table %s", len(df), table_ref)
    try:
        load_job = client.load_table_from_dataframe(
            df,
            table_ref,
            job_config=job_config,
            location=location,
        )
    except TypeError:
        load_job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    load_job.result()
    logging.info("BigQuery load complete: %s", table_ref)


def run_job() -> None:
    record_count = int(os.getenv("IFRS9_RECORD_COUNT", "5000"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "/app/output"))
    gcs_bucket = os.getenv("GCS_OUTPUT_BUCKET")
    gcs_prefix = os.getenv("GCS_OBJECT_PREFIX", "ifrs9-batch")

    bq_project = os.getenv("BQ_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT_ID")
    bq_dataset = os.getenv("BQ_DATASET")
    bq_table = os.getenv("BQ_TABLE", "loan_portfolio")
    bq_location = os.getenv("BQ_LOCATION")
    bq_write_disposition = os.getenv("BQ_WRITE_DISPOSITION", "WRITE_TRUNCATE")

    portfolio = generate_portfolio(record_count)
    summary = summarise_portfolio(portfolio)

    outputs = write_local_outputs(portfolio, summary, output_dir)
    maybe_upload_to_gcs(outputs, gcs_bucket, gcs_prefix)
    maybe_write_to_bigquery(
        portfolio,
        bq_project,
        bq_dataset,
        bq_table,
        location=bq_location,
        write_disposition=bq_write_disposition,
    )

    logging.info("Job completed successfully")


def run_healthcheck() -> int:
    try:
        _ = DataGenerator(seed=1)
        return 0
    except Exception as exc:  # pragma: no cover - safety net
        logging.exception("Healthcheck failed: %s", exc)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    args = argv if argv is not None else sys.argv[1:]
    if args and args[0] == "--healthcheck":
        return run_healthcheck()

    if _SHUTDOWN_REQUESTED:
        logging.warning("Shutdown requested before job start. Exiting.")
        return 0

    run_job()
    return 0


if __name__ == "__main__":
    sys.exit(main())
