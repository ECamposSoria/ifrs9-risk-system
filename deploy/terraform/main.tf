# IFRS9 Risk System - Production Terraform Configuration
# Main infrastructure setup for multi-environment deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
  backend "gcs" {
    bucket = var.terraform_state_bucket
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
  common_labels = {
    environment = var.environment
    project     = "ifrs9-risk-system"
    managed_by  = "terraform"
  }
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "dataproc.googleapis.com",
    "aiplatform.googleapis.com",
    "composer.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudkms.googleapis.com",
    "iam.googleapis.com"
  ])
  
  service = each.key
  project = var.project_id
  
  disable_dependent_services = false
  disable_on_destroy        = false
}

# Network infrastructure
module "vpc" {
  source = "./modules/vpc"
  
  project_id   = var.project_id
  environment  = var.environment
  region       = var.region
  labels       = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

# IAM and service accounts
module "iam" {
  source = "./modules/iam"
  
  project_id  = var.project_id
  environment = var.environment
  labels      = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

# Cloud KMS for encryption
module "kms" {
  source = "./modules/kms"
  
  project_id  = var.project_id
  environment = var.environment
  region      = var.region
  labels      = local.common_labels
  
  depends_on = [google_project_service.required_apis]
}

# Cloud Storage for data lake
module "cloud_storage" {
  source = "./modules/cloud-storage"
  
  project_id        = var.project_id
  environment       = var.environment
  region            = var.region
  labels            = local.common_labels
  kms_key_id        = module.kms.storage_key_id
  lifecycle_rules   = var.storage_lifecycle_rules
  
  depends_on = [module.kms]
}

# BigQuery data warehouse
module "bigquery" {
  source = "./modules/bigquery"
  
  project_id    = var.project_id
  environment   = var.environment
  region        = var.region
  labels        = local.common_labels
  kms_key_id    = module.kms.bigquery_key_id
  
  # Service account for data access
  service_accounts = {
    dataproc = module.iam.dataproc_service_account_email
    composer = module.iam.composer_service_account_email
    vertex   = module.iam.vertex_service_account_email
  }
  
  depends_on = [module.kms, module.iam]
}

# Dataproc for Spark processing
module "dataproc" {
  source = "./modules/dataproc"
  
  project_id            = var.project_id
  environment           = var.environment
  region                = var.region
  labels                = local.common_labels
  subnet_name           = module.vpc.subnet_name
  service_account_email = module.iam.dataproc_service_account_email
  
  # Auto-scaling configuration
  enable_autoscaling    = true
  min_workers          = var.dataproc_min_workers
  max_workers          = var.dataproc_max_workers
  
  depends_on = [module.vpc, module.iam]
}

# Vertex AI platform
module "vertex_ai" {
  source = "./modules/vertex-ai"
  
  project_id            = var.project_id
  environment           = var.environment
  region                = var.region
  labels                = local.common_labels
  service_account_email = module.iam.vertex_service_account_email
  subnet_name           = module.vpc.subnet_name
  
  depends_on = [module.vpc, module.iam]
}

# Google Kubernetes Engine
module "gke" {
  source = "./modules/gke"
  
  project_id    = var.project_id
  environment   = var.environment
  region        = var.region
  labels        = local.common_labels
  network_name  = module.vpc.network_name
  subnet_name   = module.vpc.subnet_name
  
  # Node pool configuration
  node_config = var.gke_node_config
  
  depends_on = [module.vpc]
}

# Cloud Composer for Airflow orchestration
module "composer" {
  source = "./modules/composer"
  
  project_id            = var.project_id
  environment           = var.environment
  region                = var.region
  labels                = local.common_labels
  network_name          = module.vpc.network_name
  subnet_name           = module.vpc.subnet_name
  service_account_email = module.iam.composer_service_account_email
  
  # Environment configuration
  airflow_config = var.composer_airflow_config
  
  depends_on = [module.vpc, module.iam]
}

# Monitoring and logging
module "monitoring" {
  source = "./modules/monitoring"
  
  project_id  = var.project_id
  environment = var.environment
  labels      = local.common_labels
  
  # Notification channels
  notification_emails = var.monitoring_notification_emails
  
  depends_on = [google_project_service.required_apis]
}

# Secret Manager for sensitive data
module "secrets" {
  source = "./modules/secrets"
  
  project_id  = var.project_id
  environment = var.environment
  labels      = local.common_labels
  
  # Secrets to create
  secrets = var.secret_configs
  
  depends_on = [google_project_service.required_apis]
}

# Load balancer for API services
module "load_balancer" {
  source = "./modules/load-balancer"
  
  project_id   = var.project_id
  environment  = var.environment
  labels       = local.common_labels
  backend_services = {
    ifrs9_api    = module.gke.service_endpoints.ifrs9_api
    ml_service   = module.gke.service_endpoints.ml_service
    data_service = module.gke.service_endpoints.data_service
  }
  
  # SSL certificate configuration
  ssl_certificate_domains = var.ssl_certificate_domains
  
  depends_on = [module.gke]
}

# Cloud SQL for operational data (optional)
module "cloud_sql" {
  count  = var.enable_cloud_sql ? 1 : 0
  source = "./modules/cloud-sql"
  
  project_id     = var.project_id
  environment    = var.environment
  region         = var.region
  labels         = local.common_labels
  network_name   = module.vpc.network_name
  
  # Database configuration
  database_version = var.cloud_sql_database_version
  tier            = var.cloud_sql_tier
  
  depends_on = [module.vpc]
}

# Backup and disaster recovery
module "backup" {
  source = "./modules/backup"
  
  project_id  = var.project_id
  environment = var.environment
  region      = var.region
  labels      = local.common_labels
  
  # Resources to backup
  backup_targets = {
    bigquery_datasets = module.bigquery.dataset_ids
    storage_buckets   = module.cloud_storage.bucket_names
    gke_cluster       = module.gke.cluster_name
  }
  
  # Backup schedule
  backup_schedule = var.backup_schedule
  
  depends_on = [
    module.bigquery,
    module.cloud_storage,
    module.gke
  ]
}