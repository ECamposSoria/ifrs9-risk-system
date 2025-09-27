# IFRS9 Risk System - Production Terraform Configuration
# Main infrastructure setup for multi-environment deployment

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.40"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.40"
    }
  }
  backend "gcs" {
    prefix = "terraform/state"
  }
}

# Configure providers
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Local variables
locals {
  environment = var.environment
  required_api_services = {
    compute              = "compute.googleapis.com"
    container            = "container.googleapis.com"
    bigquery             = "bigquery.googleapis.com"
    bigquerydatatransfer = "bigquerydatatransfer.googleapis.com"
    storage              = "storage.googleapis.com"
    dataproc             = "dataproc.googleapis.com"
    aiplatform           = "aiplatform.googleapis.com"
    composer             = "composer.googleapis.com"
    monitoring           = "monitoring.googleapis.com"
    logging              = "logging.googleapis.com"
    secretmanager        = "secretmanager.googleapis.com"
    cloudkms             = "cloudkms.googleapis.com"
    iam                  = "iam.googleapis.com"
    gkebackup            = "gkebackup.googleapis.com"
    storagetransfer      = "storagetransfer.googleapis.com"
    servicenetworking    = "servicenetworking.googleapis.com"
    artifactregistry     = "artifactregistry.googleapis.com"
    notebooks            = "notebooks.googleapis.com"
    cloudresourcemanager = "cloudresourcemanager.googleapis.com"
    iamcredentials       = "iamcredentials.googleapis.com"
    billingbudgets       = "billingbudgets.googleapis.com"
    sqladmin             = "sqladmin.googleapis.com"
    run                  = "run.googleapis.com"
    workflows            = "workflows.googleapis.com"
    cloudscheduler       = "cloudscheduler.googleapis.com"
  }
  bucket_base_name = replace(lower("ifrs9-${var.project_id}-${var.environment}"), "_", "-")
  storage_bucket_names = {
    raw       = "${local.bucket_base_name}-raw"
    processed = "${local.bucket_base_name}-processed"
    models    = "${local.bucket_base_name}-models"
    artifacts = "${local.bucket_base_name}-artifacts"
    backups   = "${local.bucket_base_name}-backups"
  }
  common_labels = {
    environment = var.environment
    project     = "ifrs9-risk-system"
    managed_by  = "terraform"
  }
}

resource "google_kms_crypto_key_iam_binding" "storage_encrypters" {
  crypto_key_id = module.kms.storage_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  members       = local.kms_storage_members

  depends_on = [
    module.kms,
    google_project_service.required_apis["cloudkms"],
    google_project_service.required_apis["storage"],
    google_project_service.required_apis["storagetransfer"]
  ]
}

resource "google_kms_crypto_key_iam_binding" "bigquery_encrypters" {
  crypto_key_id = module.kms.bigquery_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  members       = local.kms_bigquery_members

  depends_on = [
    module.kms,
    google_project_service.required_apis["cloudkms"],
    google_project_service.required_apis["bigquery"],
    google_project_service.required_apis["bigquerydatatransfer"]
  ]
}

resource "google_project_iam_member" "composer_service_agent_extension" {
  project = var.project_id
  role    = "roles/composer.ServiceAgentV2Ext"
  member  = format("serviceAccount:service-%s@cloudcomposer-accounts.iam.gserviceaccount.com", data.google_project.current.number)

  depends_on = [google_project_service.required_apis["composer"]]
}

resource "google_storage_bucket_iam_member" "bucket_access" {
  for_each = local.storage_bucket_member_bindings

  bucket = each.value.bucket
  role   = each.value.role
  member = each.value.member

  depends_on = [module.cloud_storage]
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = local.required_api_services

  service = each.value
  project = var.project_id

  disable_dependent_services = false
  disable_on_destroy         = false
}

data "google_project" "current" {
  project_id = var.project_id
}

resource "google_project_service_identity" "bigquery" {
  provider = google-beta
  project  = var.project_id
  service  = "bigquery.googleapis.com"

  depends_on = [google_project_service.required_apis["bigquery"]]
}

resource "google_project_service_identity" "bigquery_data_transfer" {
  provider = google-beta
  project  = var.project_id
  service  = "bigquerydatatransfer.googleapis.com"

  depends_on = [google_project_service.required_apis["bigquerydatatransfer"]]
}

resource "google_project_service_identity" "storage" {
  provider = google-beta
  project  = var.project_id
  service  = "storage.googleapis.com"

  depends_on = [google_project_service.required_apis["storage"]]
}

data "google_storage_transfer_project_service_account" "default" {
  project    = var.project_id
  depends_on = [google_project_service.required_apis["storagetransfer"]]
}

data "external" "autopilot_gk3_nodes" {
  count = var.enable_autopilot_drain_guard && var.enable_gke ? 1 : 0
  program = [
    "bash",
    "${path.module}/scripts/check_gk3_nodes.sh",
    var.project_id
  ]
}

# Network infrastructure
module "vpc" {
  source = "./modules/vpc"

  project_id                  = var.project_id
  environment                 = var.environment
  region                      = var.region
  labels                      = local.common_labels
  enforce_autopilot_drain     = var.enable_autopilot_drain_guard && var.enable_gke
  autopilot_active_nodes      = local.autopilot_active_node_count
  autopilot_active_node_names = local.autopilot_active_node_names

  depends_on = [google_project_service.required_apis["compute"]]
}

resource "google_compute_global_address" "psa_range" {
  name          = "ifrs9-${var.environment}-psa-range"
  project       = var.project_id
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = module.vpc.network_self_link

  depends_on = [google_project_service.required_apis["servicenetworking"]]
}

resource "google_service_networking_connection" "psa_connection" {
  network                 = module.vpc.network_self_link
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.psa_range.name]

  depends_on = [google_project_service.required_apis["servicenetworking"]]
}

# IAM and service accounts
module "iam" {
  source = "./modules/iam"

  project_id  = var.project_id
  environment = var.environment
  labels      = local.common_labels

  depends_on = [google_project_service.required_apis["iam"]]
}

# Cloud KMS for encryption
module "kms" {
  source = "./modules/kms"

  project_id  = var.project_id
  environment = var.environment
  region      = var.region
  labels      = local.common_labels

  depends_on = [google_project_service.required_apis["cloudkms"]]
}

# Cloud Storage for data lake
module "cloud_storage" {
  source = "./modules/cloud-storage"

  project_id            = var.project_id
  environment           = var.environment
  region                = var.region
  labels                = local.common_labels
  kms_key_id            = module.kms.storage_key_id
  lifecycle_rules       = var.storage_lifecycle_rules
  bucket_retention_days = var.storage_bucket_retention_days

  depends_on = [module.kms, google_kms_crypto_key_iam_binding.storage_encrypters, google_project_service.required_apis["storage"]]
}

# BigQuery data warehouse
module "bigquery" {
  source = "./modules/bigquery"

  project_id              = var.project_id
  environment             = var.environment
  region                  = var.region
  labels                  = local.common_labels
  kms_key_id              = module.kms.bigquery_key_id
  data_retention_policies = var.data_retention_policies
  pipeline_writers_group  = var.bigquery_pipeline_writers_group
  dashboard_viewers_group = var.bigquery_dashboard_viewers_group
  dataset_owner_email     = "ecampos@itba.edu.ar"
  enable_tables           = var.enable_bigquery_tables

  # Service account for data access
  service_accounts = {
    dataproc = module.iam.dataproc_service_account_email
    composer = module.iam.composer_service_account_email
    vertex   = module.iam.vertex_service_account_email
  }

  depends_on = [
    module.kms,
    module.iam,
    google_kms_crypto_key_iam_binding.bigquery_encrypters,
    google_project_service.required_apis["bigquery"]
  ]
}

locals {
  iam_service_accounts                  = module.iam.service_account_emails
  compute_default_service_account_email = format("%s-compute@developer.gserviceaccount.com", data.google_project.current.number)
  storage_transfer_service_account      = data.google_storage_transfer_project_service_account.default.email
}

locals {
  autopilot_guard_result      = var.enable_autopilot_drain_guard && var.enable_gke ? data.external.autopilot_gk3_nodes[0].result : {}
  autopilot_active_node_count = try(tonumber(lookup(local.autopilot_guard_result, "count", "0")), 0)
  autopilot_active_node_names = compact(split(";", lookup(local.autopilot_guard_result, "instances", "")))
}

locals {
  kms_storage_members = distinct(compact([
    try(format("serviceAccount:%s", google_project_service_identity.storage.email), null),
    try(format("serviceAccount:%s", data.google_storage_transfer_project_service_account.default.email), null),
    try(format("serviceAccount:service-%s@gs-project-accounts.iam.gserviceaccount.com", data.google_project.current.number), null),
    try(format("serviceAccount:%s", local.iam_service_accounts["dataproc"]), null),
    try(format("serviceAccount:%s", local.iam_service_accounts["composer"]), null),
    try(format("serviceAccount:%s", local.iam_service_accounts["vertex"]), null),
    try(format("serviceAccount:%s", local.iam_service_accounts["gke"]), null),
    try(format("serviceAccount:%s", local.compute_default_service_account_email), null)
  ]))
  kms_bigquery_members = distinct(compact([
    try(format("serviceAccount:%s", google_project_service_identity.bigquery.email), null),
    try(format("serviceAccount:%s", google_project_service_identity.bigquery_data_transfer.email), null),
    try(format("serviceAccount:bq-%s@bigquery-encryption.iam.gserviceaccount.com", data.google_project.current.number), null),
    try(format("serviceAccount:%s", local.iam_service_accounts["dataproc"]), null),
    try(format("serviceAccount:%s", local.iam_service_accounts["composer"]), null),
    try(format("serviceAccount:%s", local.iam_service_accounts["vertex"]), null),
    try(format("serviceAccount:%s", local.compute_default_service_account_email), null)
  ]))
}

locals {
  storage_bucket_member_bindings = {
    raw_dataproc_view = {
      bucket = module.cloud_storage.raw_data_bucket_name
      role   = "roles/storage.objectViewer"
      member = format("serviceAccount:%s", local.iam_service_accounts["dataproc"])
    }
    raw_composer_view = {
      bucket = module.cloud_storage.raw_data_bucket_name
      role   = "roles/storage.objectViewer"
      member = format("serviceAccount:%s", local.iam_service_accounts["composer"])
    }
    raw_storage_transfer_view = {
      bucket = module.cloud_storage.raw_data_bucket_name
      role   = "roles/storage.objectViewer"
      member = format("serviceAccount:%s", data.google_storage_transfer_project_service_account.default.email)
    }
    processed_dataproc_admin = {
      bucket = module.cloud_storage.processed_bucket_name
      role   = "roles/storage.objectAdmin"
      member = format("serviceAccount:%s", local.iam_service_accounts["dataproc"])
    }
    processed_composer_view = {
      bucket = module.cloud_storage.processed_bucket_name
      role   = "roles/storage.objectViewer"
      member = format("serviceAccount:%s", local.iam_service_accounts["composer"])
    }
    processed_storage_transfer_view = {
      bucket = module.cloud_storage.processed_bucket_name
      role   = "roles/storage.objectViewer"
      member = format("serviceAccount:%s", data.google_storage_transfer_project_service_account.default.email)
    }
    models_vertex_admin = {
      bucket = module.cloud_storage.models_bucket_name
      role   = "roles/storage.objectAdmin"
      member = format("serviceAccount:%s", local.iam_service_accounts["vertex"])
    }
    artifacts_gke_view = {
      bucket = module.cloud_storage.artifacts_bucket_name
      role   = "roles/storage.objectViewer"
      member = format("serviceAccount:%s", local.iam_service_accounts["gke"])
    }
    backups_storage_transfer_admin = {
      bucket = module.cloud_storage.backups_bucket_name
      role   = "roles/storage.objectAdmin"
      member = format("serviceAccount:%s", data.google_storage_transfer_project_service_account.default.email)
    }
  }
}

# Dataproc for Spark processing
module "dataproc" {
  source = "./modules/dataproc"
  count  = var.enable_dataproc ? 1 : 0

  project_id            = var.project_id
  environment           = var.environment
  region                = var.region
  labels                = local.common_labels
  subnet_name           = module.vpc.subnet_name
  service_account_email = module.iam.dataproc_service_account_email

  # Auto-scaling configuration
  enable_autoscaling = true
  min_workers        = var.dataproc_min_workers
  max_workers        = var.dataproc_max_workers

  depends_on = [module.vpc, module.iam, google_project_service.required_apis["dataproc"]]
}

# Vertex AI platform
module "vertex_ai" {
  count  = var.enable_vertex_workbench ? 1 : 0
  source = "./modules/vertex-ai"

  project_id            = var.project_id
  environment           = var.environment
  region                = var.vertex_region
  zone                  = var.vertex_zone
  labels                = local.common_labels
  service_account_email = module.iam.vertex_service_account_email
  network_self_link     = module.vpc.network_self_link
  subnet_self_link      = module.vpc.subnet_self_link

  depends_on = [
    module.vpc,
    module.iam,
    google_project_service.required_apis["aiplatform"],
    google_project_service.required_apis["notebooks"]
  ]
}

# Google Kubernetes Engine
module "gke" {
  source = "./modules/gke"
  count  = var.enable_gke ? 1 : 0

  project_id   = var.project_id
  environment  = var.environment
  region       = var.region
  labels       = local.common_labels
  network_name = module.vpc.network_name
  subnet_name  = module.vpc.subnet_name

  # Node pool configuration
  node_config = var.gke_node_config

  pod_secondary_range_name     = module.vpc.pod_secondary_range_name
  service_secondary_range_name = module.vpc.service_secondary_range_name
  node_service_account_email   = module.iam.gke_service_account_email
  enable_private_endpoint      = false
  authorized_networks          = var.network_security.authorized_networks
  enable_network_policy        = var.network_security.enable_network_policy
  enable_deletion_protection   = var.environment == "prod"

  depends_on = [module.vpc, google_service_networking_connection.psa_connection, google_project_service.required_apis["container"]]
}

locals {
  gke_cluster_id        = var.enable_gke ? module.gke[0].cluster_id : null
  gke_service_catalog   = var.enable_gke ? module.gke[0].service_catalog : {}
  gke_node_pool_name    = var.enable_gke ? module.gke[0].node_pool_name : null
  gke_cluster_available = var.enable_gke
}

# Cloud Composer for Airflow orchestration
module "composer" {
  source = "./modules/composer"
  count  = var.enable_composer ? 1 : 0

  project_id            = var.project_id
  environment           = var.environment
  region                = var.region
  labels                = local.common_labels
  network_name          = module.vpc.network_name
  subnet_name           = module.vpc.subnet_name
  service_account_email = module.iam.composer_service_account_email

  # Environment configuration
  airflow_config = var.composer_airflow_config

  depends_on = [
    module.vpc,
    module.iam,
    google_service_networking_connection.psa_connection,
    google_project_service.required_apis["composer"],
    google_project_iam_member.composer_service_agent_extension
  ]
}

# Monitoring and logging
module "monitoring" {
  source = "./modules/monitoring"

  project_id  = var.project_id
  environment = var.environment
  labels      = local.common_labels

  # Notification channels
  notification_emails          = var.monitoring_notification_emails
  audit_dataset_id             = module.bigquery.analytics_dataset_id
  billing_account_id           = var.billing_account_id
  budget_monthly_amount        = var.budget_monthly_amount
  budget_alert_spend_threshold = var.budget_alert_spend_threshold
  budget_currency_code         = var.budget_currency_code

  depends_on = [
    google_project_service.required_apis["monitoring"],
    google_project_service.required_apis["logging"],
    google_project_service.required_apis["billingbudgets"],
    module.bigquery
  ]
}

# Secret Manager for sensitive data
module "secrets" {
  source = "./modules/secrets"

  project_id  = var.project_id
  environment = var.environment
  labels      = local.common_labels

  # Secrets to create
  secrets = var.secret_configs

  depends_on = [google_project_service.required_apis["secretmanager"]]
}

# Load balancer for API services
module "load_balancer" {
  source = "./modules/load-balancer"
  count  = var.enable_gke ? 1 : 0

  project_id  = var.project_id
  environment = var.environment
  labels      = local.common_labels
  backend_services = {
    portfolio = {
      bucket_name = module.cloud_storage.artifacts_bucket_name
      description = "IFRS9 portfolio frontend"
      enable_cdn  = true
    }
  }
  default_backend = "portfolio"

  # SSL certificate configuration
  ssl_certificate_domains = var.ssl_certificate_domains

  depends_on = [module.gke, google_project_service.required_apis["compute"]]
}

# Cloud SQL for operational data (optional)
module "cloud_sql" {
  count  = var.enable_cloud_sql ? 1 : 0
  source = "./modules/cloud-sql"

  project_id   = var.project_id
  environment  = var.environment
  region       = var.region
  labels       = local.common_labels
  network_name = module.vpc.network_name

  # Database configuration
  database_version = var.cloud_sql_database_version
  tier             = var.cloud_sql_tier

  depends_on = [
    module.vpc,
    google_service_networking_connection.psa_connection,
    google_project_service.required_apis["sqladmin"]
  ]
}

# Backup and disaster recovery
module "backup" {
  source = "./modules/backup"
  count  = var.enable_backup_services && var.enable_gke ? 1 : 0

  project_id  = var.project_id
  environment = var.environment
  region      = var.region
  labels      = local.common_labels

  gke_cluster_id = local.gke_cluster_id
  gcs_backup_pairs = [
    {
      source_bucket      = local.storage_bucket_names.raw
      destination_bucket = local.storage_bucket_names.backups
    },
    {
      source_bucket      = local.storage_bucket_names.processed
      destination_bucket = local.storage_bucket_names.backups
    }
  ]

  # Backup schedule
  backup_schedule = var.backup_schedule

  depends_on = [
    module.bigquery,
    module.cloud_storage,
    module.gke,
    google_project_service.required_apis["gkebackup"],
    google_project_service.required_apis["storagetransfer"]
  ]
}

# Serverless orchestration (Cloud Run + Workflows + Scheduler)
module "serverless_orchestration" {
  source = "./modules/serverless"
  count  = var.enable_serverless_orchestration ? 1 : 0

  project_id                = var.project_id
  environment               = var.environment
  region                    = var.serverless_region
  labels                    = local.common_labels
  cloud_run_image           = var.serverless_cloud_run_image
  cloud_run_timeout_seconds = var.serverless_timeout_seconds
  schedule_cron             = var.serverless_schedule_cron

  depends_on = [
    google_project_service.required_apis["run"],
    google_project_service.required_apis["workflows"],
    google_project_service.required_apis["cloudscheduler"]
  ]
}
