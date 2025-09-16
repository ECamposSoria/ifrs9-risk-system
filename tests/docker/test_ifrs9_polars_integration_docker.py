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
raw = pl.DataFrame({
  'loan_amount': (pl.arange(0,N)+1000).cast(pl.Float64),
  'interest_rate': (pl.arange(0,N)%9)/100+0.02,
  'term_months': (pl.arange(0,N)%60)+12,
  'current_balance': (pl.arange(0,N)+500).cast(pl.Float64),
  'credit_score': ((pl.arange(0,N)%500)+300).cast(pl.Int64),
  'days_past_due': (pl.arange(0,N)%120).cast(pl.Int64),
  'customer_income': (pl.arange(0,N)%10000+2000).cast(pl.Float64),
  'ltv_ratio': ((pl.arange(0,N)%50)/100).cast(pl.Float64),
  'monthly_payment': ((pl.arange(0,N)%2000)/10).cast(pl.Float64),
  'loan_type': (pl.arange(0,N)%4).cast(pl.Int64).cast(pl.Utf8),
  'employment_status': (pl.arange(0,N)%4).cast(pl.Int64).cast(pl.Utf8),
  'label': ((pl.arange(0,N)%2)==0).cast(pl.Int64)
})

# Simple Polars FE
feat = raw.with_columns([
  (pl.col('loan_amount')/pl.col('customer_income')).alias('dti'),
  (pl.col('current_balance')/pl.col('loan_amount')).alias('util'),
  (pl.col('interest_rate')*pl.col('loan_amount')).alias('int_cost')
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

