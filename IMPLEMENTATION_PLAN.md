# (ifrs9) Portfolio Deployment — Implementation Plan (Revised)

## Last Updated: 2025-12-13
## Status: PENDING APPROVAL

---

# IMPLEMENTATION PLAN

## OBJECTIVE

Package this repo as a **portfolio-friendly, low-cost GCP deployment** that demonstrates:
1. **Repeatable deployment** (Terraform applies cleanly; destroy is possible).
2. **BigQuery-first analytics** (data lands in BigQuery; Looker Studio connects to views).
3. **Optional Vertex AI** (on-demand training + batch inference; nothing always-on).
4. **Cost guardrails** (budgets + “do not leave expensive infra running” defaults).

**Important correction:** the current `deploy/terraform` root module creates **VPC + Cloud NAT** and **CMEK (Cloud KMS)** by default. Cloud NAT is *not* free-tier and will blow the “<$1” goal if left running. This plan proposes a portfolio-safe path to avoid that.

---

## DOCS (context7)

Looked up via Context7:
- **BigQuery Python client (`/googleapis/python-bigquery`)**
  - Loading Pandas DataFrames requires `pyarrow` (`google-cloud-bigquery[pandas,pyarrow]`).
  - Datetime handling differs for naive vs tz-aware columns; keep types explicit where possible.
- **Terraform Google provider (`/hashicorp/terraform-provider-google`)**
  - `google_cloud_run_v2_job` supports container env configuration and deletion protection.
  - Use dataset-scoped IAM via `google_bigquery_dataset_iam_member` for least privilege.
- **Vertex AI Python SDK (`/googleapis/python-aiplatform`)**
  - Registering a model requires `Model.upload(...)` and a serving container image URI (prebuilt or custom).

---

## ARCHITECTURE

### Target “portfolio” architecture (no always-on inference endpoints)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              GCP PROJECT                                      │
│                        (academic-ocean-472500-j4)                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Cloud Run Job (batch)                                                      │
│   - Generates synthetic portfolio                                             │
│   - Loads to BigQuery                                                         │
│   - Optionally uploads artifacts to GCS                                        │
│            │                                                                   │
│            ▼                                                                   │
│   BigQuery                                                                    │
│   - ifrs9_raw_<env>.loan_portfolio                                             │
│   - ifrs9_analytics_<env> views for Looker                                     │
│   - (optional) ifrs9_ml_<env> predictions                                      │
│            │                                                                   │
│            ▼                                                                   │
│   Looker Studio (free)                                                        │
│   - Connects to analytics views                                                │
│                                                                              │
│   (Optional) Vertex AI                                                        │
│   - Custom training job reads BigQuery                                         │
│   - Writes model artifacts to GCS                                              │
│   - (Optional) registers model + runs batch predictions                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## PRIVILEGE REQUIREMENTS (WHAT NEEDS SUDO FROM YOU)

This VM currently has **Docker**, but **does not have** `gcloud`, `bq`, `terraform`, or `pip` installed.

If you want to run deployment from this VM, you will need to install:
- `google-cloud-cli` (for `gcloud` + `bq`)
- `terraform` (or `opentofu`)

I won’t run `sudo`; I’ll provide commands for you to run.

---

## FILES TO CREATE / MODIFY (PROPOSED)

### Modify
- `docker/cloud-run-job/cloud_run_job.py` — add BigQuery load step.
- `docker/cloud-run-job/requirements.txt` — add `google-cloud-bigquery` (and keep `pyarrow` compatible with current pins).
- `deploy/terraform/cloud-run-job.tf` — add BigQuery env vars + IAM (dataset-level + `roles/bigquery.jobUser`).
- `deploy/terraform/staging.auto.tfvars` (or a new portfolio tfvars) — align toggles for low-cost demo.

### Add
- `IMPLEMENTATION_PLAN.md` — this document.
- **Approved (separate stack)**: `deploy/portfolio/terraform/*` — minimal portfolio Terraform (no VPC/NAT/KMS).
- **Approved**: `deploy/portfolio/terraform/portfolio.auto.tfvars` — portfolio defaults.
- **Approved (Vertex training-only)**: `docker/vertex-training/*` — training container + script (writes artifacts to GCS).

---

## TECHNICAL DECISIONS

### 1) Avoid “training jobs in Terraform”
Terraform is declarative. Creating a `Vertex AI Custom Job` as a Terraform resource makes “terraform apply” run an execution workload and introduces drift.

Decision:
- Terraform provisions **infrastructure only** (APIs, service accounts, buckets, IAM).
- Training/prediction runs are triggered via `gcloud ai custom-jobs create ...` (explicit, on-demand).

### 2) BigQuery ingestion strategy
For a portfolio demo dataset (5k–50k rows), **direct load from DataFrame** is simplest and cheap.

Decision:
- Use `Client.load_table_from_dataframe(..., write_disposition=WRITE_TRUNCATE)` for deterministic “latest snapshot” dashboards.
- Add a `run_id` / `generated_at` column if we want history later.

### 3) Least-privilege IAM
Project-level `roles/bigquery.dataEditor` is too broad.

Decision:
- Grant Cloud Run job service account:
  - `roles/bigquery.jobUser` at project-level (to create load jobs).
  - `roles/bigquery.dataEditor` (or `roles/bigquery.dataOwner` if table creation needed) **at dataset-level** for `ifrs9_raw_<env>`.

### 4) Cost target realism
The original plan’s “<$1 and $0/mo” estimate is not accurate if we deploy the current Terraform root module unchanged (Cloud NAT alone is a meaningful hourly cost).

Decision:
- **Approved:** create a **separate minimal portfolio Terraform stack** in `deploy/portfolio/terraform` that does not create VPC/NAT/CMEK.

---

## CONTAINERIZATION STRATEGY

- Keep existing Cloud Run job image (`docker/cloud-run-job/Dockerfile`) — already pinned and non-root.
- Avoid `:latest` tags (policy); use semantic tags (e.g., `0.2.0`).
- Vertex training image (if enabled):
  - Pin base image (match `python:3.11.7-slim`).
  - Run as non-root.
  - Keep dependencies minimal (sklearn + BigQuery + GCS).

---

## IMPLEMENTATION STEPS (ORDERED CHECKLIST)

### Phase 0 — Preflight
- [x] Use a **separate minimal portfolio Terraform stack** (`deploy/portfolio/terraform`).
- [ ] Install tooling (if deploying from this VM): `gcloud`, `bq`, `terraform`.
- [ ] Confirm GCP auth:
  - [ ] `gcloud auth login`
  - [ ] `gcloud auth application-default login` (needed for Terraform provider auth)

### Phase 1 — BigQuery-ready data pipeline (Cloud Run Job → BigQuery)
- [ ] Update `docker/cloud-run-job/cloud_run_job.py`:
  - [ ] Add `google-cloud-bigquery` optional import.
  - [ ] Implement `write_to_bigquery(df, project, dataset, table, disposition)`.
  - [ ] Call it from `run_job()` when `BQ_*` env vars are present.
- [ ] Update `docker/cloud-run-job/requirements.txt`:
  - [ ] Add `google-cloud-bigquery` (pin compatible version range).
  - [ ] Keep `pyarrow` version consistent with the image build constraints.
- [ ] Rebuild + push Cloud Run image with a new tag (e.g., `0.2.0`).

### Phase 2 — Terraform: IAM + env wiring (no expensive defaults)
- [ ] Add env vars to `deploy/terraform/cloud-run-job.tf`:
  - `BQ_PROJECT_ID`, `BQ_DATASET`, `BQ_TABLE` (defaults: `ifrs9_raw_${environment}`, `loan_portfolio`).
- [ ] Add IAM for Cloud Run job SA:
  - [ ] `roles/bigquery.jobUser` at project-level.
  - [ ] `google_bigquery_dataset_iam_member` for dataset writer access.
- [ ] Deploy portfolio infra via `deploy/portfolio/terraform` (no NAT/CMEK by design).

### Phase 3 — Analytics layer (BigQuery views)
- [ ] Create the views via Terraform in the analytics dataset so Looker can connect immediately.
- [ ] Create/extend views required for Looker Studio:
  - `ecl_by_stage`
  - `geographic_distribution`
  - `product_analysis`
  - `risk_metrics`
  - `credit_score_bands`

### Phase 4 — Optional Vertex AI (training only)
- [ ] Add `docker/vertex-training/` (training container + `train.py`).
- [ ] Push the training image with a pinned tag.
- [ ] Run training via `gcloud ai custom-jobs create` (explicit, on-demand).

### Phase 5 — Looker Studio dashboard
- [ ] Connect Looker Studio to `ifrs9_analytics_<env>` dataset.
- [ ] Build dashboard (as per design) using analytics views.
- [ ] Record dashboard link in `README.md` + `memory.md`.

### Phase 6 — Teardown
- [ ] `terraform destroy` (portfolio profile) after capturing screenshots/links.
- [ ] Verify no always-on services remain.

---

## POTENTIAL CHALLENGES & MITIGATIONS

- **Cost blowups from network/NAT**: avoid creating Cloud NAT for portfolio, or destroy immediately after demo.
- **BigQuery module schema files missing** (`deploy/terraform/modules/bigquery/schemas/*.json` are not in the repo snapshot): keep `enable_bigquery_tables=false` for portfolio, or add the schema files before enabling.
- **Partition filter issues**: if tables are created with `require_partition_filter=true`, ensure views include partition filters (otherwise Looker queries will fail).
- **Vertex “free” assumption**: Vertex AI training and batch predictions are billable; keep them optional and bounded (timeouts + small data).

---

## COST BREAKDOWN (REVISED / REALISTIC)

Costs depend heavily on what Terraform provisions and how long it remains running.

- **Cloud Run Job**: typically pennies for a single short run (often covered by free tier).
- **BigQuery**: free tier covers small storage and queries, but verify region rules.
- **Vertex AI**: training + batch prediction are billable; run once, small machine, short timeout.
- **Cloud NAT (if created)**: meaningful hourly cost → **must be avoided for the <$1 goal**.
- **Cloud KMS (if created)**: small recurring cost per key version; avoid or accept as non-zero.

Recommendation: set a small project budget (e.g., $1–$5) with alerts before running Vertex.

---

## APPROVAL

Approved:
1. **Separate portfolio stack** (`deploy/portfolio/terraform`) rather than refactoring `deploy/terraform`.
2. **BigQuery ingestion: `WRITE_TRUNCATE`** for “latest snapshot” dashboards.
3. **Vertex AI: training only** (artifacts to GCS; no model registry, no batch inference).
