# IFRS9 Risk System Terraform Deployment Guide

This guide bridges the gap between the local Docker workflow and a managed Google Cloud Platform (GCP) deployment. It documents prerequisites, IAM expectations, and the workflow for planning and applying the Terraform configuration in `deploy/terraform`.

## Prerequisites
- **GCP project** with billing enabled.
- **Terraform state bucket** (`var.terraform_state_bucket`) created manually with versioning enabled (recommended). Grant the Terraform service account `roles/storage.objectAdmin`.
- **CLI tooling**: Terraform ≥ 1.5, gcloud ≥ 460, and application-default credentials (`gcloud auth application-default login`).
- **Service accounts** (can be created manually or by Terraform modules):
  - `ifrs9-terraform@<project>.iam.gserviceaccount.com` (Terraform runner) with `roles/editor` plus `roles/resourcemanager.projectIamAdmin` during bootstrap.
  - `ifrs9-dataproc@<project>.iam.gserviceaccount.com` with `roles/dataproc.editor`, `roles/storage.objectAdmin`, `roles/logging.logWriter`.
  - `ifrs9-composer@<project>.iam.gserviceaccount.com` with `roles/composer.worker`, `roles/storage.objectAdmin`.
  - `ifrs9-vertex@<project>.iam.gserviceaccount.com` with `roles/aiplatform.user`, `roles/storage.objectViewer`.

## Variables
Populate a `terraform.tfvars` (or use `-var` flags) with the following minimum keys:
```hcl
project_id              = "<gcp-project-id>"
region                  = "europe-west1"
environment             = "prod"
terraform_state_bucket  = "ifrs9-terraform-state"
dataproc_min_workers    = 2
dataproc_max_workers    = 6
storage_lifecycle_rules = [
  {
    action = { type = "Delete" }
    condition = { age = 90 }
  }
]
```

### Cost Control Toggles
To keep personal deployments within the Always Free allowances, toggle major components on or off via the new boolean variables:

- `enable_gke`, `enable_composer`, `enable_dataproc`, `enable_backup_services` — disable these to avoid standing up Autopilot clusters, Composer, Dataproc, and backup jobs.
- `enable_serverless_orchestration` — provision the Cloud Run + Workflows + Cloud Scheduler replacement (idle cost ≈ $0); override `serverless_region` if you must deploy outside the default `us-central1` free-tier region.
- `enable_autopilot_drain_guard` — keeps subnet updates from running while Autopilot maintenance VMs (`gk3-*`) still reference the old secondary range. Leave enabled unless you plan to manage the drain manually.
- `enable_bigquery_tables` — skip provisioned table schemas, routines, and scheduled queries when you only need datasets (handy when schema files aren’t available or you want to minimise costs).
- Override `serverless_cloud_run_image` and the `ifrs9_batch_job` block in your tfvars when you build and publish a bespoke Cloud Run container.

`staging.auto.tfvars` now showcases the free-tier profile:

```hcl
enable_gke                      = false
enable_composer                 = false
enable_dataproc                 = false
enable_backup_services          = false
enable_serverless_orchestration = true
serverless_region               = "us-central1"
enable_bigquery_tables          = false
```

## Workflow
```bash
cd deploy/terraform
terraform init \
  -backend-config="bucket=$(terraform_state_bucket)" \
  -backend-config="prefix=terraform/state" \
  -backend-config="project=<gcp-project-id>"
terraform workspace new prod || terraform workspace select prod
terraform plan -out=tfplan
terraform apply tfplan
```

#### Autopilot drain guard (GKE secondary ranges)
When `enable_autopilot_drain_guard` is left at its default (`true`), Terraform calls `scripts/check_gk3_nodes.sh` and blocks subnet updates until no Autopilot maintenance instances (`gk3-*`) remain. Monitor the drain with:

```bash
gcloud compute instances list --project <gcp-project-id> --filter="name~gk3-" --format="value(name,zone)"
```

Set the variable to `false` only when you purposely skip the guard (for example, after deleting the cluster) and accept that the subnet update may still fail if nodes linger.


### CI/CD Integration
### CI/CD Integration
- Configure a dedicated Terraform runner with the Terraform service account key stored in Secret Manager.
- Use `terraform plan -detailed-exitcode` in CI to gate merges; require manual approval for `terraform apply`.

## Post-Apply Checklist
1. Confirm enabled APIs via `gcloud services list --enabled`.
2. Verify stateful resources:
   - VPC and subnets (`gcloud compute networks subnets list`).
   - Dataproc cluster (`gcloud dataproc clusters list`).
   - BigQuery dataset `ifrs9_data_<env>` and tables.
   - GCS buckets (`gsutil ls gs://ifrs9-*`).
3. If Composer is enabled, deploy DAGs (see `terraform output composer_environment`); otherwise trigger the serverless workflow execution returned by `terraform output serverless_orchestration`.

## Next Steps to Production
- Wire application secrets into Secret Manager (`terraform output secret_manager_ids`).
- Configure workload identity federation for CI runners to replace key files.
- Update monitoring pipelines to push metrics to Cloud Monitoring (`monitoring/` stack).
- Run `make test` locally and in a Dataproc session to confirm parity before promoting workloads.
- For minimal-cost demos, leave only the serverless orchestration enabled and toggle GKE/Composer/Dataproc on just for load or integration tests.
