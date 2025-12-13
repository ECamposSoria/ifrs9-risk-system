# IFRS9 Portfolio Terraform (Minimal, Low-Cost)

This Terraform stack is a **portfolio/demo-friendly** deployment that avoids the heavy/production modules in `deploy/terraform` (notably VPC + Cloud NAT + CMEK).

It provisions:
- BigQuery datasets: `ifrs9_raw_<env>`, `ifrs9_analytics_<env>`, `ifrs9_ml_<env>`
- A GCS bucket for job/model artifacts
- A Cloud Run **Job** that generates data and loads it into BigQuery
- Optional Vertex AI **training-only** service account + permissions (no endpoints, no batch prediction)
- Dashboard-ready BigQuery **views** for Looker Studio

## Prereqs (you run)
- `gcloud` installed + authenticated
- `terraform` installed
- Billing enabled on the project

## Quickstart
```bash
cd deploy/portfolio/terraform

# Optional: copy and edit defaults
cp portfolio.auto.tfvars.example portfolio.auto.tfvars

terraform init
terraform plan -var-file=portfolio.auto.tfvars -out=tfplan
terraform apply tfplan
```

## Build + Push images (you run)
Cloud Run job image (update tag as needed):
```bash
docker build -f docker/cloud-run-job/Dockerfile -t gcr.io/<PROJECT_ID>/ifrs9-cloud-run-job:0.2.0 .
docker push gcr.io/<PROJECT_ID>/ifrs9-cloud-run-job:0.2.0
```

Vertex training image (optional):
```bash
docker build -t <REGION>-docker.pkg.dev/<PROJECT_ID>/ifrs9-ml/training:0.1.0 docker/vertex-training
docker push <REGION>-docker.pkg.dev/<PROJECT_ID>/ifrs9-ml/training:0.1.0
```

## Run the pipeline (you run)
1) Execute Cloud Run job:
```bash
gcloud run jobs execute ifrs9-portfolio-batch --region <CLOUD_RUN_REGION> --wait
```

2) Looker Studio connects to views in `ifrs9_analytics_<env>` (created by Terraform).

## Teardown
```bash
terraform destroy -var-file=portfolio.auto.tfvars
```

