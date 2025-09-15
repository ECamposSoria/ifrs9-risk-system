# System Memory

## Last Updated: 2025-09-15 18:12 UTC
## Version: 0.1.0

### Current Architecture
- Flow: synthetic data generation (`src/generate_data.py`), multi-layer validation (`src/validation.py`, `validation/` orchestration), PySpark IFRS9 rules execution (`src/rules_engine.py`), ML enrichment (`src/ml_model.py`, `src/enhanced_ml_models.py`, `src/polars_ml_integration.py`), cloud integration (`src/gcp_integrations.py`), and reporting/monitoring via Looker-ready BigQuery views (`dashboards/`) and Prometheus/Grafana (`monitoring/`).
- Orchestration: Airflow DAG (`dags/ifrs9_pipeline.py`) coordinates eight named agents defined by orchestration config (`config/orchestration_rules.yaml`) and Makefile tasks.
- Source modules: `src/ai_explanations.py` (Vertex AI/Gemini explainability), `src/analysis/gemini_codebase_analyzer.py` (offline repo scanner with optional Gemini), `src/agents/runtime_server.py` (FastAPI agent wrapper), `src/polars_ml_benchmark.py`, `src/polars_ml_integration.py`, `src/production_validation_generator.py`, `src/security/security_middleware.py`, `src/validation_simple.py`, `src/generate_data.py`, `src/gcp_integrations.py`, `src/production_validation_generator.py`.
- Configuration: `config/ifrs9_rules.yaml` drives staging thresholds, risk parameters, ECL logic, outputs, and ML toggles; `config/orchestration_rules.yaml` defines SLAs, retries, agent sequence, monitoring targets, integration endpoints, and governance checks.
- Validation ecosystem: `validation/` holds containerized validators, orchestrators, datetime converters, production readiness scripts, and summary reports; `comprehensive_system_validation.py` and `test_validation.py` provide additional end-to-end checks.
- Deployment & Ops: Dockerfiles (`Dockerfile.ifrs9-spark`, `docker/airflow/*`, `docker/jupyter/*`, `docker/agents/*`, `docker/production/*`, `docker/validation/*`), docker-compose files (`docker-compose.ifrs9.yml`, `docker_polars_test.yml`), Kubernetes manifests (`k8s/`), Helm charts (`charts/`), Argo CD definitions (`argo/`), Terraform scaffolding (`deploy/terraform/`), backup orchestration (`backup/backup_recovery_orchestrator.py`), monitoring stack (`monitoring/`), and load-testing resources (`deploy/loadtest/`, `testing/`).
- Data & analytics artifacts: `dashboards/` (Looker Studio setup + BigQuery views), `notebooks/` (EDA and pipeline notebooks), `data/` (raw/processed placeholders), `docs/` (ARCHITECTURE, DEPLOYMENT, K8S_DEPLOYMENT, USER_GUIDE), `IFRS9_PROJECT_COMPLETE_SUMMARY.md`, `IFRS9_SECURITY_AUDIT_REPORT.md`, `PRODUCTION_INFRASTRUCTURE_README.md`.
- Testing: `tests/` (unit/integration/Polars performance), `testing/` (environment/load validators), `scripts/` (agent coordination, validation runners, ML integration tests), `run_docker_validation.sh`, `run_docker_polars_tests.sh`, `run_production_validation.py`.

### Container Setup
- Base images: `python:3.10-slim` with Spark 3.5.4 (`Dockerfile.ifrs9-spark`), `apache/airflow:2.7.0-python3.10` (`docker/airflow/Dockerfile.ifrs9-airflow`), `jupyter/pyspark-notebook:python-3.10` (`docker/jupyter/Dockerfile.ifrs9-jupyter`), plus service-specific derivatives under `docker/agents/`, `docker/production/`, and `docker/validation/`.
- Core services (`docker-compose.ifrs9.yml`): `postgres` (15-alpine), `spark-master`, `spark-worker`, `jupyter`, `airflow-webserver`, `airflow-scheduler`, `airflow-init`, optional `pgadmin`; shared volumes mount `dags/`, `logs/`, `plugins/`, `data/`, `src/`, `tests/`, `notebooks/`.
- Networking: single `ifrs9-network` with exposed ports dynamically mapped; health checks on Spark UIs, Jupyter, Airflow components; environment variables configure Spark (`SPARK_MODE`, `SPARK_MASTER_URL`), Airflow core (`AIRFLOW__*`), authentication tokens (`JUPYTER_TOKEN`).
- Supporting stacks: monitoring compose (`deploy/monitoring/docker-compose.monitoring.yml`) launching Prometheus (9090), Grafana (3000), Alertmanager (9093), custom Polars exporter (9092); load-test compose (`deploy/loadtest/docker-compose.loadtest.yml`) exposing Locust UI (8089).

### Implemented Features
- IFRS9 PySpark rules engine (`src/rules_engine.py`): configurable staging with SICR detection, ML-assisted stage blending, PD/LGD/EAD calculations, ECL with discounting and macro scenarios, aggregation, audit trail, summary reporting, validation hooks. Driven by YAML config and expects ML callbacks from `src/ml_model.py` when enabled.
- Machine learning stack:
  - Baseline classifier/regressor (`src/ml_model.py`) with feature engineering, scaling, RandomForest models, PD regression, evaluation metrics, persistence helpers.
  - Advanced ensemble pipeline (`src/enhanced_ml_models.py`) using Polars-aware feature engineering, Optuna tuning, XGBoost/LightGBM/CatBoost integration, shap explainability, visualization, and health checks.
  - Polars-native workflows (`src/polars_ml_integration.py`, `src/polars_ml_benchmark.py`) delivering lazy evaluation pipelines, hybrid conversions, benchmarking utilities.
- Data validation suite: Pandera- and Great Expectations-based schemas (`src/validation.py`), simplified fallback validator (`src/validation_simple.py`), containerized validation agents (`validation/`), production validation generator (`src/production_validation_generator.py`), orchestration scripts for containerized QA (`validation/master_validation_orchestrator.py`, `validation/master_production_orchestrator.py`).
- Synthetic data generation: high-fidelity Spanish/LATAM loan data generator (`src/generate_data.py`) plus production-scale scenario builder with stress cases, multi-currency, and economic overlays (`src/production_validation_generator.py`).
- Generative AI & explanations: Vertex AI/Gemini explanation engine (`src/ai_explanations.py`) for PD predictions, anomaly detection, natural-language narratives; repo analyzer with optional Gemini summarization (`src/analysis/gemini_codebase_analyzer.py`).
- Cloud integration: BigQuery/Cloud Storage/Dataproc toolkit (`src/gcp_integrations.py`) for dataset/table management, dataframe uploads, schema enforcement, job submission; backup orchestrator with cross-cloud replication (`backup/backup_recovery_orchestrator.py`).
- Orchestration: Enhanced Airflow DAG (`dags/ifrs9_pipeline.py`) implementing agent state tracking, SLA enforcement, Slack/email notifications, GCS/BigQuery operations, recovery logic; orchestration policies stored in `config/orchestration_rules.yaml`.
- Security & runtime: FastAPI agent server with JWT, rate limiting, Prometheus metrics (`src/agents/runtime_server.py` + `src/security/security_middleware.py`), security audit report (`IFRS9_SECURITY_AUDIT_REPORT.md`).
- Monitoring & ops: Prometheus rules, Grafana dashboards, exporter scripts (`monitoring/`), production infra guide (`PRODUCTION_INFRASTRUCTURE_README.md`), Terraform/Helm/Argo templates for Kubernetes deployments (`k8s/`, `charts/`, `argo/`, `deploy/terraform/`).
- Documentation & training: Architecture/deployment/user guides (`docs/`), Looker Studio walkthrough and SQL views for analytics (`dashboards/looker_setup.md`, `dashboards/bigquery_views.sql`).

### Containerization & Deployment Gaps vs Job Posting
- Job demands AWS big data tooling (S3, Glue, Redshift); current stack is GCP-centric with only documentary mention of S3 in backups—no AWS SDK integrations beyond boto3 placeholders.
- Power BI expertise requested; dashboards target Looker Studio with no Power BI assets or export scripts.
- Multi-agent Docker images exist, but runtime services beyond Spark/Airflow/Jupyter are not wired into docker-compose; FastAPI agent images require manual build/run.

### API Endpoints
- Agent runtime server (`src/agents/runtime_server.py`) exposes `/healthz`, `/readyz`, `/metrics`, and `/run` (JWT-protected, invokes configured agent `main()`); no dedicated IFRS9 calculation REST API is implemented despite K8s manifests referencing `ifrs9-api` deployment.

### Database Schema & Analytics Assets
- BigQuery view definitions (`dashboards/bigquery_views.sql`) create reporting tables: portfolio overview, stage distribution, credit quality trends, regional risk, product performance, data quality metrics, executive summary, regulatory report.
- No relational schema migrations provided; database assumptions documented in README/PRODUCTION_INFRASTRUCTURE_README.md.

### Key Functions and Classes
- `IFRS9RulesEngine` (`src/rules_engine.py:15`): orchestrates validation, staging, risk parameter and ECL computation with ML hooks and audit logging.
- `CreditRiskClassifier` (`src/ml_model.py:22`): prepares features, trains RandomForest stage classifier and PD regressor, outputs metrics.
- `AdvancedFeatureEngineer` & `AdvancedMLPipeline` wrappers (`src/enhanced_ml_models.py`) enabling Polars-native feature creation, Optuna tuning, shap explainability.
- `PolarsEnhancedCreditRiskClassifier` (`src/polars_ml_integration.py:27`): lazy Polars processing with XGBoost/LightGBM bridging.
- `DataValidator` (`src/validation.py:18`) and simplified counterpart (`src/validation_simple.py:12`): schema/business rule validation with metrics capture.
- `ProductionValidationGenerator` (`src/production_validation_generator.py:42`): scenario templates for validation datasets and Arrow/Parquet exports.
- `GCPIntegration` + `BigQueryManager` (`src/gcp_integrations.py:20`): credential loading, dataset/table CRUD, dataframe loads, Dataproc job submission.
- `VertexAIExplanationEngine` (`src/ai_explanations.py:24`): Vertex AI clients, Gemini integration, PD prediction wrappers, SHAP explanation pipeline.
- Validation orchestrators (`validation/master_validation_orchestrator.py:20`, `validation/master_production_orchestrator.py`) coordinating dockerized agents and scoring readiness.
- Monitoring exporter (`monitoring/polars-performance-exporter.py`) and load-test harness (`testing/load_testing_framework.py`).

### Integration Points
- Google Cloud: BigQuery, Cloud Storage, Dataproc, Vertex AI (requires credentials), Cloud Monitoring endpoints configured in YAML.
- ML Libraries: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, shap, Polars.
- Workflow tooling: Apache Airflow (LocalExecutor), Spark master/worker cluster, Prometheus/Grafana, Locust load testing.
- Security: JWT via custom middleware, optional Slack/email notifications, backup to GCS/S3 (S3 usage stubbed).

### Deployment Instructions
- Local bootstrap: `make setup`, `make init-airflow`, `make up`, `make show-ports` (Makefile targets).
- Testing: `make test`, `make test-rules`, `make test-polars`, `scripts/run_production_validation.py`, `run_docker_validation.sh`.
- Monitoring stack: `make monitoring-up` / `make monitoring-down`.
- Load testing: `make loadtest-up` / `make loadtest-down` (Locust UI at 8089).
- Kubernetes/Argo/Terraform: manifests in `k8s/`, Helm charts in `charts/`, Argo CD apps in `argo/`, Terraform scaffolding in `deploy/terraform/`; production deployment script `deploy/production_deployment.py` is placeholder for automation.
- Backup/DR: run `backup/backup_recovery_orchestrator.py` with cloud credentials for scheduled backups and verification service.

### Recent Changes
- 2025-09-15: Initial system-wide inventory captured; documented architecture, container stack, feature coverage, and compliance vs job requirements.

### Known Issues
- Tests reference outdated APIs: `tests/test_rules.py` expects `_apply_staging_rules` and `_calculate_risk_parameters` while `IFRS9RulesEngine` exposes `_apply_enhanced_staging_rules`/`_calculate_enhanced_risk_parameters`; `comprehensive_system_validation.py` imports `MLModel` class that no longer exists.
- Airflow DAG and orchestration configs depend on BigQuery/Vertex AI/Slack credentials; repository provides no secrets management scripts beyond documentation.
- AWS tooling mandated by job posting (S3/Glue/Redshift ETL, IAM patterns) absent; boto3 usage limited to backup script placeholders and lacks tested implementations.
- Reporting deliverables target Looker; no Power BI datasets, PBIX templates, or export automation provided.
- Kubernetes manifests reference `ifrs9-api` image and init containers not present in repository; container registry/deployment process undefined.
- Security audit and production readiness docs claim features (distroless images, Cosign, CI scanning) that have no corresponding config or pipeline in repo.
- Validation summaries under `validation/` assert production readiness without executable evidence; several scripts rely on hardcoded `/opt/airflow` paths unsuitable for local dev.
- Generative AI integrations (`src/ai_explanations.py`, `src/analysis/gemini_codebase_analyzer.py`) require Vertex AI access and will fail offline; no stubs or configuration toggles besides environment variables.

### Next Steps
- Implement AWS data platform parity (S3 ingestion, Glue jobs, Redshift/Spectrum loaders) to satisfy job requirements and diversify cloud footprint.
- Deliver Power BI assets or export pipelines alongside existing Looker content to demonstrate BI competency.
- Reconcile code/tests: update `tests/test_rules.py`, `comprehensive_system_validation.py`, and dependent scripts to match current class interfaces; add regression coverage for ML-assisted staging paths.
- Provide operational deployment glue for agent FastAPI services (compose entries, Helm charts, CI/CD workflows) or prune unused manifests claiming nonexistent services.
- Harden documentation vs implementation: align security/infra claims with actual code (resource limits, Cosign, CI scanners) or add the missing automation.
- Add credential management guidance (.env templates, Secret Manager integration) and safer defaults for Vertex AI/GCP access.
- Expand data quality controls to feed actionable metrics into dashboards and share them with data quality teams, tying into orchestration config thresholds.
