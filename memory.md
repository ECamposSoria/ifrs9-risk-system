# System Memory

## Last Updated: 2025-09-27 20:37 UTC
## Version: 0.8.1

### Current Architecture
- Terraform staging stack under `deploy/terraform` now enables required Google APIs, wires Service Networking/NAT/IAM, and supports cost-control feature flags (`enable_gke`, `enable_composer`, `enable_dataproc`, `enable_backup_services`, `enable_serverless_orchestration`, `enable_autopilot_drain_guard`, `enable_bigquery_tables`) to pivot between full and Always Free footprints.
- Serverless orchestration module couples Cloud Run, Workflows, and Cloud Scheduler (defaulting to `us-central1`) so Always Free environments can run on-demand jobs without Composer or Dataproc.
- Cloud Run batch container (`docker/cloud-run-job/`) generates synthetic portfolios, writes stage summaries, and can publish results to GCS for scheduled jobs.
- Airflow DAG `dags/ifrs9_pipeline.py` orchestrates synthetic data creation, feature engineering, IFRS9 rule staging, ML scoring, and reporting end-to-end without external agents.
- Synthetic loan generation (`src/generate_data.py:DataGenerator`) now balances provision stages, applies safe default PDs, and seeds all libraries for deterministic test fixtures.
- Dual processing paths (`src/polars_ml_integration.py`, `src/enhanced_ml_models.py`, `src/ml_model.py`) keep Polars and Pandas feature pipelines in lockstep, with streaming predictions able to operate directly on raw inputs.
- IFRS9 rules engine (`src/rules_engine.py:IFRS9RulesEngine`) drives staging, SICR detection, ECL attribution, and configurable defaults via `config/ifrs9_rules.yaml`, exposing compatibility helpers for optimized test flows.
- Supporting services (Spark, Postgres, Airflow UI, Jupyter) run through `docker-compose.ifrs9.yml`, sharing source, config, reports, validation assets, and test suites via named volumes.

### Container Setup
- Base Images: `python:3.10-slim` (Spark workers/master), Airflow/Jupyter custom images extending pinned Python runtimes; Postgres `15-alpine` for orchestration metadata.
- Services: Postgres, Spark master/worker, Airflow webserver/scheduler/triggerer/init, Jupyter Lab, validation and reporting helpers (see `docker-compose.ifrs9.yml`).
- Ports: Postgres `5432`, Spark master `7077/8080`, Spark worker UI `8081`, Airflow webserver `8080`, Jupyter `8888` (all mapped via environment overrides).
- Volumes: Shared mounts for `./src`, `./tests`, `./validation`, `./scripts`, `./config`, `./data`, `./logs`, and persisted `./reports` artifacts across containers; Postgres uses `postgres-db-volume`.
- Environment Variables: `IFRS9_RULES_CONFIG` (optional override), Spark auth flags, `SPARK_MASTER`, `JUPYTER_TOKEN`; test fixtures force `spark.sql.execution.arrow.pyspark.enabled=false` to avoid Arrow IPC issues.

### Implemented Features
- Terraform cost-control toggles for GKE/Composer/Dataproc/backups, autopilot drain guard script, optional BigQuery table provisioning, and the serverless orchestration module are now baseline features.
- Balanced synthetic stage assignment with deterministic seeds and safe probability defaults to stabilize regression tests.
- Advanced feature engineering refactored for Polars/Pandas parity, simplified model selection (RandomForest baseline), and resilient handling of preprocessed feature sets.
- IFRS9 rules engine enhancements covering SICR thresholds, optional columns with defaults, deterministic staging adjustments, Stage 2 exposure caps, and compatibility alias methods (`_apply_staging_rules`, `calculate_risk_parameters`, etc.).
- SHAP fallback gracefully uses feature importances when SHAP calculations fail, keeping explainability consistent across pipelines.
- Docker helpers, Spark fixtures, and configs hardened to run without Arrow, ensure reports directory availability, and mount updated volumes for validation artifacts.
- Streaming predictions path supports raw loan inputs while maintaining consistent accuracy between legacy and optimized pipelines.
- Test harness suppresses noisy sklearn category/precision warnings so suite output stays actionable during health runs.

### API Endpoints (if applicable)
- None — batch/stream processing is orchestrated via Airflow and Spark jobs rather than HTTP endpoints.

### Database Schema (if applicable)
- Postgres stores Airflow metadata; IFRS9 computation remains in-memory (Pandas/Polars DataFrames) with no persistent application schema tracked.

### Key Functions/Classes
- `src/generate_data.py:DataGenerator.generate_loan_portfolio` — emits balanced stage data with deterministic defaults for testing.
- `src/enhanced_ml_models.py:AdvancedFeatureEngineer` — harmonizes feature construction across Polars/Pandas with guardrails for missing inputs.
- `src/enhanced_ml_models.py:OptimizedMLPipeline` — orchestrates preprocessing, RandomForest baseline fitting, SHAP/feature-importance explanations, and streaming-safe predictions.
- `src/rules_engine.py:IFRS9RulesEngine` — central staging/SICR/ECL logic with validation and summary helpers aligned to optimized test expectations.
- `src/polars_ml_integration.py:PolarsMLOrchestrator` (and streaming utilities) — executes Polars-native workflows while mirroring Pandas outputs.

### Integration Points
- Polars/Pandas dual support for feature pipelines and model scoring.
- PySpark containers for distributed validation workloads with Arrow disabled for stability.
- SHAP explainability with feature-importance fallback.
- Docker-based orchestration (Airflow, Spark, Postgres, Jupyter) for local and CI validation.

### Deployment Instructions
- Build containers: `docker-compose -f docker-compose.ifrs9.yml build`.
- Run stack: `docker-compose -f docker-compose.ifrs9.yml up -d`.
- Execute full validation suite inside Spark container: `make test` (ensures Airflow/Spark services running).
- Infrastructure prep: from `deploy/terraform`, run `terraform init -backend=false && terraform validate`; Stage 1 applies should target `google_project_service.*` before full plan.
- Local CLI prerequisites (installed): `gcloud` with application-default login and the HashiCorp `terraform` binary.

### Recent Changes
- 2025-09-27: Updated staging tfvars with Cloud Run image/env overrides, pushed `gcr.io/academic-ocean-472500-j4/ifrs9-cloud-run-job:0.1.0`, created `ifrs9-batch-job`, and executed a verification run via `gcloud run jobs execute --wait`.
- 2025-10-27: Added BigQuery schema assets/SQL, re-enabled dataset provisioning behind `enable_bigquery_tables`, built Cloud Run batch Docker artefacts, and captured the serverless-only Terraform plan (`plan-staging.tfplan`).
- 2025-10-27: Added Terraform feature flags (GKE/Composer/Dataproc/backup/serverless), Autopilot drain guard precondition, Cloud Run + Workflows orchestration module, and updated staging defaults for Always Free alignment.
- 2025-09-25: Refined Terraform apply flow: added lazy handling for service-identity emails in KMS bindings, restored required Dataproc/Storage IAM roles, normalized Composer defaults (larger SQL CIDR, lighter overrides), disabled private GKE master endpoint, and updated Vertex Workbench base image to `tf-latest-cpu` so staging apply succeeds.
- 2025-09-24: Hardened staging Terraform configuration—pinned provider v5.40, expanded API enablement (incl. Service Networking, Billing Budgets, SQL Admin), added Cloud NAT + default-deny egress, CMEK KMS bindings via service identities, Storage/BigQuery scoped IAM, Storage Transfer permissions, HTTPS redirect-ready load balancer, monitoring budget + audit sink, and validated with `terraform validate`.
- 2025-09-18: Replaced Terraform placeholder modules for GKE, Composer, Vertex AI Workbench, load balancer, and backup orchestration; targeted staging resources to southamerica-east1, regenerated `deploy/terraform/plan-staging.txt`, and captured new state after successful `terraform plan`.
- 2025-09-18: Captured staging Terraform plan (`deploy/terraform/plan-staging.txt`) after bootstrapping core modules; GKE/Composer/Vertex/load balancer remain placeholders pending final infra design.
- 2025-09-18: Added foundational Terraform modules, sanitized backend config, ran staging init/plan, and saved `deploy/terraform/plan-staging.txt` (note placeholder modules for GKE/Composer/Vertex/Load Balancer awaiting full definitions).
- 2025-09-18: Installed Google Cloud CLI 481 locally, created \`gs://ifrs9-terraform-state-staging\`, added \`deploy/terraform/staging.auto.tfvars\`, and captured Terraform init failure due to missing local modules.
- Implemented broad stability fixes across synthetic data, feature engineering, optimized ML pipeline, and IFRS9 rules engine for predictable staging and ECL outputs.
- Added compatibility alias methods and deterministic stage ordering to satisfy optimized integration tests.
- Hardened Docker helpers, Spark fixtures, and compose volumes; disabled PySpark Arrow during tests to prevent IPC faults.
- Extended SHAP handling with feature-importance fallback and ensured streaming predictions operate on raw inputs.
- Updated `config/ifrs9_rules.yaml`, ensured `reports/.gitkeep`, and aligned Polars/Pandas accuracy.
- Added cloud-mode GCP integration regression tests (BigQuery + Cloud Storage stubs) alongside local fallbacks, lifting `src/gcp_integrations.py` coverage to ~45% and exercising failure paths (Conflict, NotFound, load/query errors).
- Refactored Docker-marked suites to support local execution without host Docker, added eager Polars constructs, and ensured `make test` now runs 87 tests with zero skips.
- Captured infrastructure guidance in `deploy/terraform/README.md`, documenting Terraform workflow, IAM expectations, and post-apply validation.
- Suppressed residual sklearn user warnings and Polars deprecations across fixtures/tests to keep health runs noise-free.
- Tests executed: `make test` (full suite via Spark container) ✔️ — 87 passed, 0 skipped.
- Expanded GCP integration logic with idempotent BigQuery helpers, retry-aware stubs, and a Dataproc local registry to support fallback verification.
- Tightened Docker fallback harness (shared volume prep, synthetic Spark env) and added parity tests to keep CI without Docker representative.
- Removed sklearn warning suppressors by normalising categorical pipelines and preserving feature metadata, eliminating "unknown categories"/feature-name messages.

### Known Issues
- Continue monitoring deterministic stage caps to ensure they align with IFRS9 policy nuances in downstream analytics.
- SHAP fallback uses feature importances; review long-term for parity with full SHAP when computational issues are resolved.

### Next Steps
- Schedule the `ifrs9-batch-job` (Cloud Scheduler or Workflow cron step) so it runs automatically with the new container.
- Review Terraform plan output and apply when ready to codify the Cloud Run job and other staged changes.
- Monitor the newly created GCS bucket (`ifrs9-batch-artifacts`) and adjust retention/permissions once artefact requirements are finalised.
- Before re-enabling GKE, confirm `scripts/check_gk3_nodes.sh` reports zero Autopilot instances to avoid subnet conflicts.
- When production workloads resume full scale, re-enable Composer/Dataproc/GKE via tfvars and revisit NAT/logging budgets.
