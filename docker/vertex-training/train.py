#!/usr/bin/env python3
"""Vertex AI training job (training-only) for IFRS9 stage prediction."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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


def load_data_from_bigquery(project_id: str, dataset: str, table: str) -> pd.DataFrame:
    client = bigquery.Client(project=project_id)
    query = f"""
    SELECT
      {", ".join(FEATURE_COLUMNS)},
      provision_stage
    FROM `{project_id}.{dataset}.{table}`
    WHERE provision_stage IS NOT NULL
    """
    logger.info("Loading training data from BigQuery: %s.%s.%s", project_id, dataset, table)
    df = client.query(query).to_dataframe()
    logger.info("Loaded %d rows", len(df))
    return df


def train_model(df: pd.DataFrame) -> Tuple[RandomForestClassifier, Dict[str, float], Dict[str, object]]:
    if df.empty:
        raise ValueError("No training rows returned from BigQuery query")

    X = df[FEATURE_COLUMNS].fillna(0)
    y = df["provision_stage"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    logger.info("Training RandomForestClassifier (%d train rows, %d test rows)", len(X_train), len(X_test))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    metrics: Dict[str, object] = {
        "classification_report": report,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    return model, importance, metrics


def upload_artifacts(
    *,
    bucket_name: str,
    model_name: str,
    run_id: str,
    model: RandomForestClassifier,
    feature_importance: Dict[str, float],
    metrics: Dict[str, object],
) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    tmp_dir = Path("/tmp") / "ifrs9-vertex-training"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model_path = tmp_dir / "model.joblib"
    metadata_path = tmp_dir / "metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framework": "scikit-learn",
        "model_type": "RandomForestClassifier",
        "feature_columns": FEATURE_COLUMNS,
        "feature_importance": feature_importance,
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    base_prefix = f"models/{model_name}/{run_id}"

    model_blob = bucket.blob(f"{base_prefix}/model.joblib")
    metadata_blob = bucket.blob(f"{base_prefix}/metadata.json")

    logger.info("Uploading model to gs://%s/%s", bucket_name, model_blob.name)
    model_blob.upload_from_filename(str(model_path))

    logger.info("Uploading metadata to gs://%s/%s", bucket_name, metadata_blob.name)
    metadata_blob.upload_from_filename(str(metadata_path))

    return f"gs://{bucket_name}/{base_prefix}/"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--bq-dataset", required=True)
    parser.add_argument("--bq-table", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--model-name", default="ifrs9_stage_model")
    args = parser.parse_args(argv)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    df = load_data_from_bigquery(args.project_id, args.bq_dataset, args.bq_table)
    model, importance, metrics = train_model(df)
    artifact_uri = upload_artifacts(
        bucket_name=args.output_bucket,
        model_name=args.model_name,
        run_id=run_id,
        model=model,
        feature_importance=importance,
        metrics=metrics,
    )

    logger.info("Training complete. Artifacts: %s", artifact_uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

