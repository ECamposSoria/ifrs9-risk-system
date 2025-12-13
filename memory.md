# System Memory

## Last Updated: 2025-12-13
## Version: 0.9.0

---

## Project Overview

**IFRS9 Risk Management System** - A production-grade financial risk calculation system implementing International Financial Reporting Standard 9 for credit risk assessment, Expected Credit Loss (ECL) calculation, and regulatory compliance.

- **Status**: Production-ready (98% completion)
- **Current Branch**: master
- **GCP Project**: `academic-ocean-472500-j4`
- **Primary Region**: `southamerica-east1`

---

## Directory Structure

```
/home/eze/projects/ifrs9-risk-system/
├── src/                          # Core Python modules
│   ├── rules_engine.py           # IFRS9 PySpark rules processing
│   ├── enhanced_ml_models.py     # Advanced ML (XGBoost/LightGBM/CatBoost)
│   ├── polars_ml_integration.py  # Polars high-performance ML
│   ├── ml_model.py               # Legacy ML classifiers
│   ├── generate_data.py          # Synthetic loan portfolio generator
│   ├── validation.py             # Pandera/Great Expectations validation
│   ├── gcp_integrations.py       # GCP BigQuery/GCS/Dataproc integration
│   ├── ai_explanations.py        # Rule-based explainability (offline)
│   ├── validation_simple.py      # Simple validation utilities
│   ├── polars_ml_benchmark.py    # Performance benchmarking
│   ├── analysis/
│   │   └── gemini_codebase_analyzer.py  # Offline codebase analysis
│   └── security/
│       └── security_middleware.py       # Security middleware
│
├── deploy/                       # Infrastructure as Code
│   └── terraform/                # GCP Terraform deployment
│       ├── main.tf               # Provider + API enablement
│       ├── cloud-run-job.tf      # Cloud Run batch job
│       ├── staging.auto.tfvars   # Staging environment config
│       ├── variables.tf          # Variable definitions
│       ├── outputs.tf            # Output definitions
│       └── modules/              # 15 Terraform modules
│           ├── gke/              # Google Kubernetes Engine
│           ├── composer/         # Cloud Composer (Airflow)
│           ├── dataproc/         # Cloud Dataproc (Spark)
│           ├── bigquery/         # BigQuery data warehouse
│           ├── cloud-storage/    # GCS buckets
│           ├── cloud-sql/        # Cloud SQL (Postgres)
│           ├── kms/              # Cloud KMS encryption
│           ├── iam/              # Identity & Access Management
│           ├── vpc/              # Virtual Private Cloud
│           ├── load-balancer/    # HTTPS load balancing
│           ├── monitoring/       # Cloud Monitoring
│           ├── backup/           # Backup orchestration
│           ├── vertex-ai/        # Vertex AI Workbench
│           └── serverless/       # Cloud Run + Workflows
│
├── docker/                       # Container definitions
│   ├── cloud-run-job/            # Cloud Run batch container
│   │   ├── Dockerfile            # Multi-stage build
│   │   ├── cloud_run_job.py      # Entrypoint script
│   │   └── requirements.txt
│   ├── airflow/
│   │   └── Dockerfile.ifrs9-airflow
│   ├── jupyter/
│   │   └── Dockerfile.ifrs9-jupyter
│   ├── validation/
│   │   └── Dockerfile.validation
│   └── production/
│       └── Dockerfile.api
│
├── config/                       # Configuration files
│   ├── ifrs9_rules.yaml          # IFRS9 staging/ECL parameters
│   └── orchestration_rules.yaml  # Airflow DAG config
│
├── dags/                         # Apache Airflow DAGs
│   └── ifrs9_pipeline.py         # Main IFRS9 orchestration
│
├── tests/                        # Test suite (14 modules)
│   ├── conftest.py               # Pytest fixtures & Spark config
│   ├── test_basic.py             # Basic unit tests
│   ├── test_rules.py             # IFRS9 rules engine tests
│   ├── test_rules_optimized.py   # Optimized rules tests
│   ├── test_polars_*.py          # Polars integration tests
│   ├── test_data_factory.py      # Data generation tests
│   ├── test_gcp_*.py             # GCP integration tests
│   └── docker/                   # Docker-specific tests
│
├── validation/                   # Validation framework
│   ├── datetime_converter.py
│   ├── docker_environment_validator.py
│   ├── polars_health_check.py
│   ├── production_readiness_validator.py
│   └── end_to_end_pipeline_validator.py
│
├── k8s/                          # Kubernetes manifests
│   ├── namespace.yaml
│   ├── deployments/
│   └── gke/
│
├── monitoring/                   # Monitoring setup
├── argo/                         # ArgoCD workflows
├── notebooks/                    # Jupyter notebooks
│   ├── EDA.ipynb
│   └── LocalPipeline.ipynb
├── data/                         # Data directories
│   ├── raw/
│   └── processed/
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
├── reports/                      # Generated reports
│
├── docker-compose.ifrs9.yml      # Multi-service orchestration
├── Dockerfile.ifrs9-spark        # Spark base image
├── Makefile                      # Build automation
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest configuration
└── .github/workflows/            # CI/CD pipelines
    ├── ci.yml                    # Unit tests, linting
    ├── ci-cd.yaml                # Additional CI/CD
    └── cd-production.yml         # Production deployment
```

---

## Key Functions/Classes

### Core IFRS9 Processing
- **`src/rules_engine.py:IFRS9RulesEngine`** — PySpark-based rules processor for staging, SICR detection, ECL calculation. Configuration-driven via `config/ifrs9_rules.yaml`.

### Data Generation
- **`src/generate_data.py:DataGenerator`** — Synthetic loan portfolio creation with balanced stage assignment, deterministic seeds, 13+ product types.

### Machine Learning
- **`src/enhanced_ml_models.py:AdvancedFeatureEngineer`** — 50+ feature engineering with Polars/Pandas parity.
- **`src/enhanced_ml_models.py:OptimizedMLPipeline`** — Model orchestration (RandomForest, XGBoost, LightGBM, CatBoost), SHAP explainability, streaming predictions.

### High-Performance Processing
- **`src/polars_ml_integration.py:PolarsMLOrchestrator`** — Polars-native workflows with 10x+ performance, lazy evaluation, streaming support.

### GCP Integration
- **`src/gcp_integrations.py:GCPIntegration`** — BigQuery, Cloud Storage, Dataproc integration with local fallback for development.

---

## Container Setup

### Base Images
- **Spark**: `python:3.10-slim` with Java 21, PySpark 3.5.4, Hadoop 3
- **Cloud Run**: `python:3.11.7` multi-stage, non-root user
- **Postgres**: `15-alpine` for Airflow metadata

### Services (`docker-compose.ifrs9.yml`)
| Service | Image | Port(s) | Purpose |
|---------|-------|---------|---------|
| postgres | postgres:15-alpine | 5432 | Airflow metadata DB |
| spark-master | custom | 7077, 8080 | Spark coordinator |
| spark-worker | custom | 8081 | Distributed processing |
| airflow-webserver | custom | 8080 | Pipeline UI |
| airflow-scheduler | custom | — | Task scheduler |
| airflow-triggerer | custom | — | Async trigger |
| jupyter | custom | 8888 | Development |
| validation | custom | — | Test harness |

### Volumes
- Shared mounts: `./src`, `./tests`, `./validation`, `./scripts`, `./config`, `./data`, `./logs`, `./reports`
- Postgres data: `postgres-db-volume`

### Environment Variables
- `IFRS9_RULES_CONFIG` — Optional rules override
- `SPARK_MASTER` — Spark master URL
- `JUPYTER_TOKEN` — Jupyter authentication
- Arrow disabled in tests: `spark.sql.execution.arrow.pyspark.enabled=false`

---

## Terraform Infrastructure

### Feature Flags (`staging.auto.tfvars`)
```hcl
enable_gke                      = false  # K8s cluster
enable_composer                 = false  # Cloud Composer
enable_dataproc                 = false  # Spark cluster
enable_backup_services          = false  # Backup jobs
enable_serverless_orchestration = false  # Cloud Run + Workflows
enable_bigquery_tables          = true   # BigQuery datasets
```

### Service Accounts
- `ifrs9-terraform` — Terraform runner
- `ifrs9-dataproc` — Spark cluster
- `ifrs9-composer` — Airflow
- `ifrs9-vertex` — AI/ML
- `ifrs9-staging-job` — Cloud Run batch

### Key Resources
- State Backend: `gs://ifrs9-terraform-state-staging`
- BigQuery Dataset: `ifrs9_data_staging`
- Cloud Run Image: `gcr.io/academic-ocean-472500-j4/ifrs9-cloud-run-job:0.1.0`
- KMS Key Ring: `ifrs9-staging-keys` (versions scheduled for destruction)

---

## CI/CD Pipelines

### `ci.yml` — Continuous Integration
- **Triggers**: Push (main/develop), Pull Requests
- **Python**: 3.10, 3.11
- **Jobs**: Code quality (Black, isort, Flake8, MyPy, Bandit), Unit tests (pytest, 80%+ coverage), Docker build, Security scanning (CodeQL, Trivy)

### `cd-production.yml` — Production Deployment
- GCP authentication
- Terraform apply with approval gate
- Cloud Run deployment
- Post-deployment validation

---

## IFRS9 Compliance Features

- **3-Stage Classification**: Stage 1 (DPD<30), Stage 2 (DPD 30-90), Stage 3 (DPD≥90)
- **SICR Detection**: 2.0x PD multiplier, 100-point credit score decline threshold
- **ECL Calculation**: PD × LGD × EAD with 2.5% risk-free discounting
- **Time Horizons**: 12-month ECL (Stage 1), lifetime ECL (Stage 2/3)
- **Forward-Looking**: Economic scenario weighting
- **Audit Trail**: 7-year retention for regulatory compliance

---

## Deployment Instructions

### Local Development
```bash
# Build containers
docker-compose -f docker-compose.ifrs9.yml build

# Start stack
docker-compose -f docker-compose.ifrs9.yml up -d

# Run tests
make test
```

### Terraform Deployment
```bash
cd deploy/terraform
terraform init -reconfigure -backend-config="bucket=ifrs9-terraform-state-staging"
terraform plan -var-file=staging.auto.tfvars
terraform apply
```

### Prerequisites
- `gcloud` CLI v549.0.1 (includes `bq` 2.1.25, `gsutil` 5.35)
- `terraform` v1.14.2
- Docker Engine 29.x+
- Docker Compose v2.40+

---

## Current Infrastructure Status

As of 2025-09-28:
- Infrastructure is **fully torn down** via targeted destroy
- Only KMS key ring `ifrs9-staging-keys` remains (versions in `DESTROY_SCHEDULED` state)
- GCP project `academic-ocean-472500-j4` is clean and ready for Infrastructure Manager bootstrap
- All AI tooling is strictly offline (Gemini/Claude hooks removed)

---

## Changelog

### 2025-12-13
- **Installed CLI tools**: Google Cloud SDK v549.0.1 (`gcloud`, `bq` 2.1.25, `gsutil` 5.35) and Terraform v1.14.2 via official APT repositories.
- Added `IMPLEMENTATION_PLAN.md` (portfolio deployment plan revision focused on BigQuery-first + optional Vertex AI).
- Added Cloud Run batch BigQuery load support in `docker/cloud-run-job/cloud_run_job.py` (controlled via `BQ_*` env vars) and dependency in `docker/cloud-run-job/requirements.txt`.
- Added minimal portfolio Terraform stack under `deploy/portfolio/terraform` (BigQuery datasets + views, GCS artifacts bucket, Cloud Run Job, optional Vertex training IAM) to avoid VPC/NAT/KMS costs from `deploy/terraform`.
- Added Vertex training-only container assets under `docker/vertex-training` (trains stage model from BigQuery and uploads artifacts to GCS).
- Flagged portfolio cost risks in existing Terraform root (`deploy/terraform`): VPC module provisions Cloud NAT by default; KMS resources are `prevent_destroy`.

### 2025-10-27
- Added BigQuery schema assets/SQL, re-enabled dataset provisioning behind `enable_bigquery_tables`
- Built Cloud Run batch Docker artifacts
- Captured serverless-only Terraform plan (`plan-staging.tfplan`)
- Added Terraform feature flags (GKE/Composer/Dataproc/backup/serverless)
- Autopilot drain guard precondition added
- Cloud Run + Workflows orchestration module

### 2025-09-28
- Extensive repository cleanup — all Gemini/Claude hooks removed
- Vertex AI integration stripped from explanations
- `.gitignore` updated for Terraform artifacts
- Terraform targeted destroy: BigQuery datasets, GCS buckets, IAM service accounts, VPC, Cloud Run job removed

### 2025-09-27
- Updated staging tfvars with Cloud Run image/env overrides
- Pushed `gcr.io/academic-ocean-472500-j4/ifrs9-cloud-run-job:0.1.0`
- Created `ifrs9-batch-job` and verified execution

### 2025-09-25
- Refined Terraform apply flow with lazy KMS bindings
- Restored Dataproc/Storage IAM roles
- Normalized Composer defaults
- Disabled private GKE master endpoint
- Updated Vertex Workbench base image

### 2025-09-24
- Hardened staging Terraform configuration
- Pinned provider v5.40
- Expanded API enablement
- Added Cloud NAT + default-deny egress
- CMEK KMS bindings via service identities

---

## Known Issues

- Continue monitoring deterministic stage caps for IFRS9 policy alignment
- SHAP fallback uses feature importances; review for full SHAP parity when computational issues resolved
- Arrow IPC disabled in Spark tests to prevent failures
- Portfolio deployment prerequisites: `pip` not installed (use if needed); `gcloud`/`bq`/`terraform` now available.
- Terraform BigQuery module references schema files not present in this repo snapshot (`deploy/terraform/modules/bigquery/schemas/*.json`); `enable_bigquery_tables=true` will fail unless those are added.
- Terraform VPC module provisions Cloud NAT by default; avoid for portfolio cost targets unless you destroy immediately.

---

## Next Steps

1. Authenticate gcloud: `gcloud auth login && gcloud auth application-default login && gcloud config set project academic-ocean-472500-j4`
2. Build + push the updated Cloud Run job image (tag `0.2.0`) and update `deploy/portfolio/terraform/portfolio.auto.tfvars` if you change the tag.
3. Deploy the portfolio stack via `deploy/portfolio/terraform` and verify BigQuery datasets/views exist.
4. Execute the Cloud Run job and confirm `ifrs9_raw_<env>.loan_portfolio` is populated (then connect Looker Studio to `ifrs9_analytics_<env>` views).
5. (Optional) Build + push `docker/vertex-training` and run a single Vertex custom training job to generate model artifacts in the portfolio GCS bucket.

---

## Test Suite

- **Test Count**: 87 tests (0 skipped)
- **Coverage Target**: 80%+
- **Markers**: `spark`, `slow`, `integration`, `unit`, `validation`, `ml`, `docker`
- **Timeout**: 300 seconds per test
- **Command**: `make test` (runs via Spark container)
