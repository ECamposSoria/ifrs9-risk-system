# IFRS9 Risk Management System

[![CI/CD Pipeline](https://github.com/your-org/ifrs9-risk-system/workflows/CI/badge.svg)](https://github.com/your-org/ifrs9-risk-system/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade IFRS9 credit risk management system implementing Expected Credit Loss (ECL) calculation, ML-based stage prediction, and analytics dashboards on Google Cloud Platform.

## Quick Links

- [View Static Results](#-static-results-no-deployment-required) - Exported data, model, and dashboard
- [Reproduce on GCP](#-reproduce-on-gcp) - Full deployment instructions
- [Local Development](#-local-development) - Docker-based setup

---

## Overview

This system demonstrates a complete **Data Engineering + Data Science + Analytics** pipeline for IFRS9 compliance:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Data Generator │───▶│  Cloud Run Job   │───▶│   BigQuery      │───▶│ Looker Studio│
│  (Synthetic)    │    │  (Batch Load)    │    │   (Analytics)   │    │  (Dashboard) │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────┘
                                                       │
                                                       ▼
                                               ┌──────────────────┐
                                               │   Vertex AI      │
                                               │  (ML Training)   │
                                               └──────────────────┘
                                                       │
                                                       ▼
                                               ┌──────────────────┐
                                               │ Batch Predictions│
                                               │  (Stage + Probs) │
                                               └──────────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Generation | Python + Cloud Run | 5000 synthetic loans with IFRS9 staging |
| Storage | BigQuery | 3 datasets (raw, analytics, ml) + 5 views |
| ML Training | Vertex AI + scikit-learn | RandomForest stage classifier (~94% accuracy) |
| Predictions | Batch job | ML predictions vs rules-based comparison |
| Visualization | Looker Studio | 6 interactive dashboards |
| Infrastructure | Terraform + Infrastructure Manager | Fully automated GCP deployment |

---

## Static Results (No Deployment Required)

All results have been exported and committed to this repository. You can explore them without deploying anything.

### Exported Data

| File | Description | Size |
|------|-------------|------|
| [`data/exports/loan_portfolio.csv`](data/exports/loan_portfolio.csv) | 5000 synthetic loans with IFRS9 staging | 1.4 MB |
| [`data/exports/predictions.csv`](data/exports/predictions.csv) | ML predictions + probabilities per stage | 599 KB |
| [`data/exports/model.joblib`](data/exports/model.joblib) | Trained RandomForest model | 5.1 MB |
| [`data/exports/metadata.json`](data/exports/metadata.json) | Training metadata (accuracy, features) | 1.6 KB |

### Dashboard PDF

The Looker Studio dashboard has been exported as a PDF:

- [`docs/Informe_ifrs9.pdf`](docs/Informe_ifrs9.pdf) - Complete dashboard with all visualizations

### Load the Model Locally

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("data/exports/model.joblib")

# Load sample data
df = pd.read_csv("data/exports/loan_portfolio.csv")

# Feature columns used for prediction
FEATURES = [
    "loan_amount", "interest_rate", "term_months", "credit_score",
    "days_past_due", "dti_ratio", "ltv_ratio", "employment_length"
]

# Make predictions
X = df[FEATURES].fillna(0)
predictions = model.predict(X)
probabilities = model.predict_proba(X)

print(f"Model type: {type(model).__name__}")
print(f"Predictions shape: {predictions.shape}")
print(f"Stage distribution: {pd.Series(predictions).value_counts().to_dict()}")
```

### Explore Predictions

```python
import pandas as pd

# Load predictions with ML probabilities
predictions = pd.read_csv("data/exports/predictions.csv")

# Compare ML vs Rules-based staging
comparison = predictions.groupby(["original_stage", "ml_predicted_stage"]).size().unstack(fill_value=0)
print("ML vs Rules Confusion Matrix:")
print(comparison)

# Check prediction confidence
print("\nAverage probability by predicted stage:")
for stage in [1, 2, 3]:
    col = f"ml_prob_stage_{stage}"
    if col in predictions.columns:
        avg_prob = predictions[predictions["ml_predicted_stage"] == stage][col].mean()
        print(f"  Stage {stage}: {avg_prob:.2%}")
```

---

## Reproduce on GCP

Follow these steps to deploy the full infrastructure and reproduce all results.

### Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated
- `terraform` >= 1.5.0 installed
- Docker installed (for building containers)

### 1. Authenticate with GCP

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Build and Push Docker Images

```bash
# Cloud Run batch job (data generation + BigQuery load)
docker build -t gcr.io/YOUR_PROJECT_ID/ifrs9-cloud-run-job:0.2.1 \
  -f docker/cloud-run-job/Dockerfile .
docker push gcr.io/YOUR_PROJECT_ID/ifrs9-cloud-run-job:0.2.1

# Vertex AI training container
docker build -t gcr.io/YOUR_PROJECT_ID/ifrs9-vertex-training:0.1.0 \
  -f docker/vertex-training/Dockerfile .
docker push gcr.io/YOUR_PROJECT_ID/ifrs9-vertex-training:0.1.0
```

### 3. Update Terraform Variables

Edit `deploy/portfolio/terraform/portfolio.auto.tfvars`:

```hcl
project_id          = "YOUR_PROJECT_ID"
region              = "southamerica-east1"      # or your preferred region
bigquery_location   = "southamerica-east1"
cloud_run_region    = "us-central1"             # Cloud Run region
environment         = "staging"

cloud_run_job_image = "gcr.io/YOUR_PROJECT_ID/ifrs9-cloud-run-job:0.2.1"

ifrs9_record_count = 5000
ifrs9_seed         = 42

enable_vertex_training_iam = true
```

### 4. Deploy Infrastructure

**Option A: Using GCP Infrastructure Manager (recommended)**

```bash
cd deploy/portfolio/terraform

# Create tarball for Infrastructure Manager
tar -czvf ../portfolio-terraform.tar.gz .

# Upload to GCS
gsutil cp ../portfolio-terraform.tar.gz gs://YOUR_BUCKET/terraform/

# Create deployment via Infrastructure Manager
gcloud infra-manager deployments apply ifrs9-portfolio-staging \
  --location=us-central1 \
  --service-account=projects/YOUR_PROJECT_ID/serviceAccounts/YOUR_SA@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --git-source-repo=gs://YOUR_BUCKET/terraform/portfolio-terraform.tar.gz \
  --input-values=project_id=YOUR_PROJECT_ID
```

**Option B: Direct Terraform**

```bash
cd deploy/portfolio/terraform
terraform init
terraform plan -var-file=portfolio.auto.tfvars
terraform apply -var-file=portfolio.auto.tfvars
```

### 5. Run Data Generation Job

```bash
gcloud run jobs execute ifrs9-portfolio-batch-staging \
  --region=us-central1 \
  --wait
```

Verify data in BigQuery:
```bash
bq query --use_legacy_sql=false \
  "SELECT COUNT(*) as row_count FROM \`YOUR_PROJECT_ID.ifrs9_raw_staging.loan_portfolio\`"
```

### 6. Train ML Model on Vertex AI

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="ifrs9-stage-training" \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=gcr.io/YOUR_PROJECT_ID/ifrs9-vertex-training:0.1.0 \
  --args="--project-id=YOUR_PROJECT_ID" \
  --args="--bq-dataset=ifrs9_raw_staging" \
  --args="--bq-table=loan_portfolio" \
  --args="--output-bucket=YOUR_PROJECT_ID-ifrs9-portfolio-artifacts-staging" \
  --args="--output-prefix=models/ifrs9_stage_model"
```

### 7. Run Batch Predictions

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="ifrs9-batch-predictions" \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=gcr.io/YOUR_PROJECT_ID/ifrs9-vertex-training:0.1.0 \
  --args="predict.py" \
  --args="--project-id=YOUR_PROJECT_ID" \
  --args="--model-bucket=YOUR_PROJECT_ID-ifrs9-portfolio-artifacts-staging" \
  --args="--model-prefix=models/ifrs9_stage_model" \
  --args="--bq-dataset=ifrs9_raw_staging" \
  --args="--output-dataset=ifrs9_analytics_staging"
```

### 8. Connect Looker Studio

1. Go to [Looker Studio](https://lookerstudio.google.com/)
2. Create new report
3. Add BigQuery data sources:
   - `ifrs9_analytics_staging.ecl_by_stage`
   - `ifrs9_analytics_staging.risk_metrics`
   - `ifrs9_analytics_staging.geographic_distribution`
   - `ifrs9_analytics_staging.product_analysis`
   - `ifrs9_analytics_staging.credit_score_bands`
   - `ifrs9_analytics_staging.loan_portfolio_predictions`

### 9. Export Results

```bash
# Export CSVs
mkdir -p data/exports

bq query --use_legacy_sql=false --format=csv --max_rows=10000 \
  "SELECT * FROM \`YOUR_PROJECT_ID.ifrs9_raw_staging.loan_portfolio\`" \
  > data/exports/loan_portfolio.csv

bq query --use_legacy_sql=false --format=csv --max_rows=10000 \
  "SELECT * FROM \`YOUR_PROJECT_ID.ifrs9_analytics_staging.loan_portfolio_predictions\`" \
  > data/exports/predictions.csv

# Download model artifacts
gsutil cp gs://YOUR_PROJECT_ID-ifrs9-portfolio-artifacts-staging/models/ifrs9_stage_model/*/model.joblib data/exports/
gsutil cp gs://YOUR_PROJECT_ID-ifrs9-portfolio-artifacts-staging/models/ifrs9_stage_model/*/metadata.json data/exports/
```

### 10. Clean Up (Avoid Costs)

```bash
# Delete via Infrastructure Manager
gcloud infra-manager deployments delete ifrs9-portfolio-staging \
  --location=us-central1

# Or via Terraform
cd deploy/portfolio/terraform
terraform destroy -var-file=portfolio.auto.tfvars
```

---

## Local Development

For local development and testing without GCP.

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM
- 10GB+ free disk space

### Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/ifrs9-risk-system.git
cd ifrs9-risk-system

# Build and start services
make setup
make up

# Check service ports
make show-ports
```

### Services

| Service | Port | Credentials |
|---------|------|-------------|
| Jupyter Lab | Dynamic | Token: `ifrs9` |
| Airflow UI | Dynamic | `airflow` / `airflow` |
| Spark Master UI | Dynamic | - |

### Run Tests

```bash
make test
```

---

## Project Structure

```
ifrs9-risk-system/
├── src/                              # Core Python modules
│   ├── rules_engine.py               # IFRS9 PySpark rules
│   ├── generate_data.py              # Synthetic data generator
│   ├── enhanced_ml_models.py         # Advanced ML (XGBoost/LightGBM)
│   └── gcp_integrations.py           # GCP BigQuery/GCS integration
│
├── docker/
│   ├── cloud-run-job/                # Cloud Run batch container
│   │   ├── Dockerfile
│   │   ├── cloud_run_job.py          # Data generation + BQ load
│   │   └── requirements.txt
│   └── vertex-training/              # Vertex AI container
│       ├── Dockerfile
│       ├── train.py                  # Model training script
│       ├── predict.py                # Batch prediction script
│       └── requirements.txt
│
├── deploy/
│   └── portfolio/terraform/          # Minimal GCP infrastructure
│       ├── main.tf                   # Provider + APIs
│       ├── bigquery.tf               # Datasets + tables
│       ├── views.tf                  # Analytics views
│       ├── cloud-run-job.tf          # Batch job
│       ├── storage.tf                # GCS bucket
│       └── portfolio.auto.tfvars     # Configuration
│
├── data/
│   └── exports/                      # Exported results
│       ├── loan_portfolio.csv        # Raw loan data
│       ├── predictions.csv           # ML predictions
│       ├── model.joblib              # Trained model
│       └── metadata.json             # Training metadata
│
├── docs/
│   └── Informe_ifrs9.pdf             # Dashboard export
│
├── tests/                            # Test suite (87 tests)
├── notebooks/                        # Jupyter notebooks
├── config/                           # Configuration files
└── memory.md                         # Project memory/changelog
```

---

## IFRS9 Implementation

### Stage Classification

| Stage | Criteria | ECL Horizon |
|-------|----------|-------------|
| Stage 1 | Performing (DPD < 30) | 12-month ECL |
| Stage 2 | Underperforming (30 <= DPD < 90) | Lifetime ECL |
| Stage 3 | Non-performing (DPD >= 90) | Lifetime ECL |

### ECL Formula

```
ECL = PD × LGD × EAD

Where:
- PD:  Probability of Default (credit score + DPD based)
- LGD: Loss Given Default (collateral adjusted)
- EAD: Exposure at Default (outstanding balance)
```

### ML Model

- **Algorithm**: RandomForestClassifier (100 estimators)
- **Features**: loan_amount, interest_rate, term_months, credit_score, days_past_due, dti_ratio, ltv_ratio, employment_length
- **Target**: provision_stage (1, 2, or 3)
- **Accuracy**: ~94% on test set

---

## Resources

- [IFRS9 Standard](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/)
- [Google Cloud BigQuery](https://cloud.google.com/bigquery/docs)
- [Vertex AI Custom Training](https://cloud.google.com/vertex-ai/docs/training/custom-training)
- [Looker Studio](https://lookerstudio.google.com/)

---

## License

MIT License - see LICENSE file for details.

## Author

**ECamposSoria**
Contact: ecampossoria88@gmail.com

---

**Last Updated**: December 2024
**Version**: 1.0.0
