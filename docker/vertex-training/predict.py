#!/usr/bin/env python3
"""Batch prediction job: loads trained model from GCS and writes predictions to BigQuery."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from google.cloud import bigquery, storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "loan_amount",
    "interest_rate",
    "term_months",
    "credit_score",
    "days_past_due",
    "dti_ratio",
    "ltv_ratio",
    "employment_length",
]


def download_model(bucket_name: str, model_prefix: str) -> Path:
    """Download the latest model from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Find the latest model run
    blobs = list(bucket.list_blobs(prefix=f"{model_prefix}/"))
    model_blobs = [b for b in blobs if b.name.endswith("model.joblib")]

    if not model_blobs:
        raise FileNotFoundError(f"No model.joblib found in gs://{bucket_name}/{model_prefix}/")

    # Sort by name (timestamp in path) and get latest
    latest_blob = sorted(model_blobs, key=lambda b: b.name, reverse=True)[0]

    local_path = Path("/tmp/model.joblib")
    logger.info("Downloading model from gs://%s/%s", bucket_name, latest_blob.name)
    latest_blob.download_to_filename(str(local_path))

    return local_path


def load_data_from_bigquery(project_id: str, dataset: str, table: str) -> pd.DataFrame:
    """Load loan data for prediction."""
    client = bigquery.Client(project=project_id)
    query = f"""
    SELECT
      loan_id,
      {", ".join(FEATURE_COLUMNS)},
      provision_stage as original_stage
    FROM `{project_id}.{dataset}.{table}`
    """
    logger.info("Loading data from BigQuery: %s.%s.%s", project_id, dataset, table)
    df = client.query(query).to_dataframe()
    logger.info("Loaded %d rows", len(df))
    return df


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """Make predictions using the trained model."""
    X = df[FEATURE_COLUMNS].fillna(0)

    logger.info("Making predictions on %d rows", len(df))
    df["ml_predicted_stage"] = model.predict(X)

    # Get prediction probabilities for each stage
    probas = model.predict_proba(X)
    classes = model.classes_

    for i, stage in enumerate(classes):
        df[f"ml_prob_stage_{stage}"] = probas[:, i]

    df["prediction_timestamp"] = datetime.now(timezone.utc)

    return df


def write_predictions_to_bigquery(
    df: pd.DataFrame,
    project_id: str,
    dataset: str,
    table: str = "loan_portfolio_predictions",
) -> None:
    """Write predictions to BigQuery."""
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset}.{table}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    logger.info("Writing %d predictions to %s", len(df), table_ref)
    load_job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    load_job.result()
    logger.info("Predictions written successfully")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--model-bucket", required=True, help="GCS bucket with trained model")
    parser.add_argument("--model-prefix", default="models/ifrs9_stage_model", help="GCS prefix for model")
    parser.add_argument("--bq-dataset", required=True, help="Source dataset with loan_portfolio")
    parser.add_argument("--bq-table", default="loan_portfolio", help="Source table")
    parser.add_argument("--output-dataset", help="Output dataset (defaults to source dataset)")
    parser.add_argument("--output-table", default="loan_portfolio_predictions")
    args = parser.parse_args(argv)

    output_dataset = args.output_dataset or args.bq_dataset

    # Download and load model
    model_path = download_model(args.model_bucket, args.model_prefix)
    model = joblib.load(model_path)
    logger.info("Model loaded: %s", type(model).__name__)

    # Load data
    df = load_data_from_bigquery(args.project_id, args.bq_dataset, args.bq_table)

    # Predict
    df_predictions = predict(model, df)

    # Write results
    write_predictions_to_bigquery(
        df_predictions,
        args.project_id,
        output_dataset,
        args.output_table,
    )

    # Log summary
    stage_counts = df_predictions["ml_predicted_stage"].value_counts().to_dict()
    logger.info("Prediction summary: %s", stage_counts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
