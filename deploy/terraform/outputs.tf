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
    name           = module.vpc.network_name
    self_link      = module.vpc.network_self_link
    subnet_name    = module.vpc.subnet_name
    subnet_cidr    = module.vpc.subnet_cidr
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
    raw_data      = module.cloud_storage.raw_data_bucket_name
    processed     = module.cloud_storage.processed_bucket_name
    models        = module.cloud_storage.models_bucket_name
    artifacts     = module.cloud_storage.artifacts_bucket_name
    backups       = module.cloud_storage.backups_bucket_name
  }
}

# BigQuery datasets
output "bigquery_datasets" {
  description = "BigQuery dataset information"
  value = {
    raw_dataset        = module.bigquery.raw_dataset_id
    processed_dataset  = module.bigquery.processed_dataset_id
    analytics_dataset  = module.bigquery.analytics_dataset_id
    ml_dataset        = module.bigquery.ml_dataset_id
  }
}

# Dataproc cluster
output "dataproc_cluster" {
  description = "Dataproc cluster information"
  value = {
    cluster_name = module.dataproc.cluster_name
    master_instance_name = module.dataproc.master_instance_name
    spark_history_server_url = module.dataproc.spark_history_server_url
  }
}

# Vertex AI platform
output "vertex_ai" {
  description = "Vertex AI platform information"
  value = {
    workbench_instance_name = module.vertex_ai.workbench_instance_name
    pipeline_service_account = module.vertex_ai.pipeline_service_account
    model_registry_location = module.vertex_ai.model_registry_location
  }
}

# GKE cluster
output "gke_cluster" {
  description = "GKE cluster information"
  value = {
    cluster_name     = module.gke.cluster_name
    cluster_endpoint = module.gke.cluster_endpoint
    cluster_ca_certificate = module.gke.cluster_ca_certificate
    kubeconfig_command = "gcloud container clusters get-credentials ${module.gke.cluster_name} --region ${var.region} --project ${var.project_id}"
  }
  sensitive = true
}

# Cloud Composer
output "composer_environment" {
  description = "Cloud Composer environment information"
  value = {
    environment_name = module.composer.environment_name
    airflow_uri     = module.composer.airflow_uri
    gcs_bucket      = module.composer.gcs_bucket
    dag_folder      = module.composer.dag_folder
  }
}

# KMS encryption keys
output "kms_keys" {
  description = "Cloud KMS encryption keys"
  value = {
    storage_key   = module.kms.storage_key_id
    bigquery_key  = module.kms.bigquery_key_id
    compute_key   = module.kms.compute_key_id
  }
  sensitive = false
}

# Load balancer
output "load_balancer" {
  description = "Load balancer information"
  value = {
    external_ip      = module.load_balancer.external_ip
    ssl_certificate  = module.load_balancer.ssl_certificate
    backend_services = module.load_balancer.backend_services
  }
}

# Monitoring
output "monitoring" {
  description = "Monitoring and alerting information"
  value = {
    notification_channels = module.monitoring.notification_channels
    alerting_policies    = module.monitoring.alerting_policies
    dashboard_urls       = module.monitoring.dashboard_urls
  }
}

# Cloud SQL (if enabled)
output "cloud_sql" {
  description = "Cloud SQL instance information"
  value = var.enable_cloud_sql ? {
    instance_name        = module.cloud_sql[0].instance_name
    connection_name      = module.cloud_sql[0].connection_name
    private_ip_address   = module.cloud_sql[0].private_ip_address
    database_names       = module.cloud_sql[0].database_names
  } : null
}

# Connection strings and endpoints
output "connection_strings" {
  description = "Connection strings for external access"
  value = {
    bigquery_project = var.project_id
    dataproc_endpoint = "${var.region}-dataproc.googleapis.com"
    vertex_ai_endpoint = "${var.region}-aiplatform.googleapis.com"
    monitoring_endpoint = "monitoring.googleapis.com"
    storage_endpoint = "storage.googleapis.com"
  }
}

# Environment-specific configurations
output "environment_config" {
  description = "Environment-specific configuration summary"
  value = {
    resource_sizing = var.resource_sizing[var.environment]
    data_retention = var.data_retention_policies[var.environment]
    backup_config = var.backup_schedule
    network_security = var.network_security
  }
}

# Deployment information
output "deployment_info" {
  description = "Deployment information and next steps"
  value = {
    terraform_version = ">=1.0"
    deployment_timestamp = timestamp()
    next_steps = [
      "1. Configure kubectl: ${local.kubeconfig_command}",
      "2. Deploy application manifests: kubectl apply -f k8s/",
      "3. Upload Airflow DAGs to: ${module.composer.gcs_bucket}/dags/",
      "4. Configure monitoring dashboards",
      "5. Set up CI/CD pipeline secrets"
    ]
  }
}

# Security and compliance
output "security_config" {
  description = "Security configuration summary"
  value = {
    kms_encryption_enabled = true
    private_networks_enabled = var.network_security.enable_private_nodes
    audit_logs_enabled = var.compliance_settings.enable_audit_logs
    network_policies_enabled = var.network_security.enable_network_policy
    iam_service_accounts_created = length(module.iam.service_account_emails)
  }
}

# Cost optimization information
output "cost_optimization" {
  description = "Cost optimization features enabled"
  value = {
    preemptible_nodes = var.gke_node_config.preemptible
    storage_lifecycle_rules = length(var.storage_lifecycle_rules)
    auto_scaling_enabled = {
      dataproc = true
      gke = true
    }
    resource_quotas = var.resource_sizing[var.environment]
  }
}

# API endpoints for applications
output "api_endpoints" {
  description = "API endpoints for application integration"
  value = {
    bigquery = "https://bigquery.googleapis.com/bigquery/v2/projects/${var.project_id}"
    vertex_ai = "https://${var.region}-aiplatform.googleapis.com"
    storage = "https://storage.googleapis.com/storage/v1/b"
    monitoring = "https://monitoring.googleapis.com/v1/projects/${var.project_id}"
    logging = "https://logging.googleapis.com/v2/projects/${var.project_id}"
  }
}

# Local variables for output processing
locals {
  kubeconfig_command = "gcloud container clusters get-credentials ${module.gke.cluster_name} --region ${var.region} --project ${var.project_id}"
}