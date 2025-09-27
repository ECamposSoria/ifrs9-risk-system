# IFRS9 Risk System - Terraform Outputs
# Resource endpoints, IDs, and connection information

# Project information
output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP region"
  value       = var.region
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

# Network outputs
output "vpc_network" {
  description = "VPC network information"
  value = {
    name        = module.vpc.network_name
    self_link   = module.vpc.network_self_link
    subnet_name = module.vpc.subnet_name
    subnet_cidr = module.vpc.subnet_cidr
    nat = {
      name            = module.vpc.nat_name
      region          = module.vpc.nat_region
      logging_enabled = module.vpc.nat_logging_enabled
    }
  }
}

# IAM service accounts
output "service_accounts" {
  description = "Service account emails for different services"
  value = {
    dataproc = module.iam.dataproc_service_account_email
    composer = module.iam.composer_service_account_email
    vertex   = module.iam.vertex_service_account_email
    gke      = module.iam.gke_service_account_email
  }
  sensitive = false
}

# Cloud Storage buckets
output "storage_buckets" {
  description = "Cloud Storage bucket information"
  value = {
    raw_data  = module.cloud_storage.raw_data_bucket_name
    processed = module.cloud_storage.processed_bucket_name
    models    = module.cloud_storage.models_bucket_name
    artifacts = module.cloud_storage.artifacts_bucket_name
    backups   = module.cloud_storage.backups_bucket_name
  }
}

# BigQuery datasets
output "bigquery_datasets" {
  description = "BigQuery dataset information"
  value = {
    raw_dataset       = module.bigquery.raw_dataset_id
    processed_dataset = module.bigquery.processed_dataset_id
    analytics_dataset = module.bigquery.analytics_dataset_id
    ml_dataset        = module.bigquery.ml_dataset_id
  }
}

# Dataproc cluster
output "dataproc_cluster" {
  description = "Dataproc cluster information"
  value = var.enable_dataproc ? {
    cluster_name             = module.dataproc[0].cluster_name
    master_instance_name     = module.dataproc[0].master_instance_name
    spark_history_server_url = module.dataproc[0].spark_history_server_url
  } : null
}

# Vertex AI platform
output "vertex_ai" {
  description = "Vertex AI platform information"
  value = {
    workbench_instance_name  = var.enable_vertex_workbench ? module.vertex_ai[0].workbench_instance_name : null
    pipeline_service_account = var.enable_vertex_workbench ? module.vertex_ai[0].pipeline_service_account : module.iam.vertex_service_account_email
    workbench_location       = var.enable_vertex_workbench ? module.vertex_ai[0].workbench_location : null
    model_registry_location  = var.vertex_region
  }
}

# GKE cluster
output "gke_cluster" {
  description = "GKE cluster information"
  value = var.enable_gke ? {
    cluster_name           = module.gke[0].cluster_name
    cluster_endpoint       = module.gke[0].cluster_endpoint
    cluster_ca_certificate = module.gke[0].cluster_ca_certificate
    kubeconfig_command     = local.kubeconfig_command
  } : null
  sensitive = true
}

# Cloud Composer
output "composer_environment" {
  description = "Cloud Composer environment information"
  value = var.enable_composer ? {
    environment_name = module.composer[0].environment_name
    airflow_uri      = module.composer[0].airflow_uri
    gcs_bucket       = module.composer[0].gcs_bucket
    dag_folder       = module.composer[0].dag_folder
  } : null
}

# KMS encryption keys
output "kms_keys" {
  description = "Cloud KMS encryption keys"
  value = {
    storage_key  = module.kms.storage_key_id
    bigquery_key = module.kms.bigquery_key_id
    compute_key  = module.kms.compute_key_id
  }
  sensitive = false
}

# Load balancer
output "load_balancer" {
  description = "Load balancer information"
  value = var.enable_gke ? {
    external_ip             = module.load_balancer[0].external_ip
    certificate_name        = module.load_balancer[0].certificate_name
    certificate_expire_time = module.load_balancer[0].certificate_expire_time
    backend_bucket_names    = module.load_balancer[0].backend_bucket_names
    http_rule               = module.load_balancer[0].http_forwarding_rule
    https_rule              = module.load_balancer[0].https_forwarding_rule
  } : null
}

output "serverless_orchestration" {
  description = "Serverless orchestration stack details"
  value = var.enable_serverless_orchestration ? {
    workflow_name         = module.serverless_orchestration[0].workflow_name
    cloud_run_service_url = module.serverless_orchestration[0].cloud_run_service_url
    scheduler_job_name    = module.serverless_orchestration[0].scheduler_job_name
  } : null
}

output "cloud_run_batch_job" {
  description = "Cloud Run batch job information"
  value = var.enable_serverless_orchestration ? {
    job_name = google_cloud_run_v2_job.ifrs9_batch[0].name
    region   = google_cloud_run_v2_job.ifrs9_batch[0].location
  } : null
}

# Monitoring
output "monitoring" {
  description = "Monitoring and alerting information"
  value = {
    notification_channel_ids = module.monitoring.notification_channel_ids
    budget_names             = module.monitoring.budget_names
    audit_log_sink           = module.monitoring.audit_log_sink
  }
}

# Cloud SQL (if enabled)
output "cloud_sql" {
  description = "Cloud SQL instance information"
  value = var.enable_cloud_sql ? {
    instance_name      = module.cloud_sql[0].instance_name
    connection_name    = module.cloud_sql[0].connection_name
    private_ip_address = module.cloud_sql[0].private_ip_address
    database_names     = module.cloud_sql[0].database_names
  } : null
}

# Connection strings and endpoints
output "connection_strings" {
  description = "Connection strings for external access"
  value = {
    bigquery_project    = var.project_id
    dataproc_endpoint   = "${var.region}-dataproc.googleapis.com"
    vertex_ai_endpoint  = "${var.vertex_region}-aiplatform.googleapis.com"
    monitoring_endpoint = "monitoring.googleapis.com"
    storage_endpoint    = "storage.googleapis.com"
  }
}

# Environment-specific configurations
output "environment_config" {
  description = "Environment-specific configuration summary"
  value = {
    resource_sizing  = var.resource_sizing[var.environment]
    data_retention   = var.data_retention_policies[var.environment]
    backup_config    = var.backup_schedule
    network_security = var.network_security
  }
}

# Deployment information
output "deployment_info" {
  description = "Deployment information and next steps"
  value = {
    terraform_version    = ">=1.0"
    deployment_timestamp = timestamp()
    next_steps           = local.deployment_next_steps
  }
}

# Security and compliance
output "security_config" {
  description = "Security configuration summary"
  value = {
    kms_encryption_enabled       = true
    private_networks_enabled     = var.network_security.enable_private_nodes
    audit_logs_enabled           = var.compliance_settings.enable_audit_logs
    network_policies_enabled     = var.network_security.enable_network_policy
    iam_service_accounts_created = length(module.iam.service_account_emails)
  }
}

# Cost optimization information
output "cost_optimization" {
  description = "Cost optimization features enabled"
  value = {
    preemptible_nodes       = var.gke_node_config.preemptible
    storage_lifecycle_rules = length(var.storage_lifecycle_rules)
    auto_scaling_enabled = {
      dataproc   = var.enable_dataproc
      gke        = var.enable_gke
      serverless = var.enable_serverless_orchestration
    }
    resource_quotas = var.resource_sizing[var.environment]
  }
}

# API endpoints for applications
output "api_endpoints" {
  description = "API endpoints for application integration"
  value = {
    bigquery   = "https://bigquery.googleapis.com/bigquery/v2/projects/${var.project_id}"
    vertex_ai  = "https://${var.vertex_region}-aiplatform.googleapis.com"
    storage    = "https://storage.googleapis.com/storage/v1/b"
    monitoring = "https://monitoring.googleapis.com/v1/projects/${var.project_id}"
    logging    = "https://logging.googleapis.com/v2/projects/${var.project_id}"
  }
}

# Local variables for output processing
locals {
  kubeconfig_command = var.enable_gke ? "gcloud container clusters get-credentials ${module.gke[0].cluster_name} --region ${var.region} --project ${var.project_id}" : "GKE cluster disabled"
  deployment_next_steps = [
    var.enable_gke ? "1. Configure kubectl: ${local.kubeconfig_command}" : "1. GKE disabled - enable var.enable_gke to configure kubectl",
    var.enable_gke ? "2. Deploy application manifests: kubectl apply -f k8s/" : "2. Application manifests deployment deferred until GKE is enabled",
    var.enable_composer ? "3. Upload Airflow DAGs to: ${module.composer[0].dag_folder}" : "3. Cloud Composer disabled - use serverless orchestration workflow",
    "4. Configure monitoring dashboards",
    "5. Set up CI/CD pipeline secrets"
  ]
}
