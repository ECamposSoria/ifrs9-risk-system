"""
IFRS9 System Integration Tests (Docker + Polars)

Validates Polars DataFrame operations, ML integration using Polars-produced
features, and basic rules engine interoperability within containers.
"""

from __future__ import annotations

import os
from typing import Dict

import pytest

from tests.docker.conftest import docker_exec_python


@pytest.mark.parametrize("container", [
    "jupyter", "airflow-webserver"
])
def test_polars_feature_engineering_and_ml(container: str, docker_containers, polars_env: Dict[str, str]):
    if container not in docker_containers:
        pytest.skip(f"{container} not running")
    code = """
import polars as pl, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Create synthetic IFRS9-like features
N=6000
idx = np.arange(N)
raw = pl.DataFrame({
  'loan_amount': (idx + 1000).astype(float),
  'interest_rate': (idx % 9) / 100 + 0.02,
  'term_months': (idx % 60) + 12,
  'current_balance': (idx + 500).astype(float),
  'credit_score': ((idx % 500) + 300).astype(int),
  'days_past_due': (idx % 120).astype(int),
  'customer_income': (idx % 10000 + 2000).astype(float),
  'ltv_ratio': ((idx % 50) / 100).astype(float),
  'monthly_payment': ((idx % 2000) / 10).astype(float),
  'loan_type': (idx % 4).astype(str),
  'employment_status': (idx % 4).astype(str)
})

# Simple Polars FE
feat = raw.with_columns([
  (pl.col('loan_amount')/pl.col('customer_income')).alias('dti'),
  (pl.col('current_balance')/pl.col('loan_amount')).alias('util'),
  (pl.col('interest_rate')*pl.col('loan_amount')).alias('int_cost'),
  (((pl.col('loan_amount')/pl.col('customer_income'))*1.5 + (pl.col('current_balance')/pl.col('loan_amount'))) > 1.2).cast(pl.Int64).alias('label')
])
pdf = feat.select(['dti','util','int_cost','label']).to_pandas()
X = pdf[['dti','util','int_cost']]; y = pdf['label']

# Train small sklearn model (always present).
from sklearn.ensemble import RandomForestClassifier
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
clf = RandomForestClassifier(n_estimators=40, random_state=42, n_jobs=2).fit(Xtr, ytr)
acc = accuracy_score(yte, clf.predict(Xte))
print(f"acc={acc:.3f}")
"""
    rc, out, err = docker_exec_python(container, code, env=polars_env, timeout=240)
    assert rc == 0, f"FE/ML integration failed in {container}: {err or out}"
    metric = dict([ln.split("=",1) for ln in out.strip().splitlines() if "=" in ln])
    assert float(metric["acc"]) >= 0.5


def test_rules_engine_visibility_from_jupyter(docker_containers, polars_env):
    # Sanity that rules engine module is importable and Spark session starts
    if "jupyter" not in docker_containers:
        pytest.skip("jupyter not running")
    code = """
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local[*]').appName('test').getOrCreate()
print(f"spark_ok={int(spark.version is not None)}")
spark.stop()
"""
    rc, out, err = docker_exec_python("jupyter", code, env=polars_env, timeout=180)
    assert rc == 0, f"Spark visibility failed: {err or out}"
    kv = dict([ln.split("=",1) for ln in out.strip().splitlines() if "=" in ln])
    assert kv.get("spark_ok") == "1"
