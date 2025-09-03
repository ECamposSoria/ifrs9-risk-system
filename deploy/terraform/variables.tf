# IFRS9 Risk System - Terraform Variables
# Environment-specific configuration variables

# Core project variables
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "europe-west1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "terraform_state_bucket" {
  description = "GCS bucket for Terraform state"
  type        = string
}

# Dataproc configuration
variable "dataproc_min_workers" {
  description = "Minimum number of Dataproc workers"
  type        = number
  default     = 2
}

variable "dataproc_max_workers" {
  description = "Maximum number of Dataproc workers"
  type        = number
  default     = 10
}

# GKE configuration
variable "gke_node_config" {
  description = "GKE node pool configuration"
  type = object({
    machine_type   = string
    disk_size_gb   = number
    disk_type      = string
    min_node_count = number
    max_node_count = number
    preemptible    = bool
  })
  default = {
    machine_type   = "e2-standard-4"
    disk_size_gb   = 100
    disk_type      = "pd-standard"
    min_node_count = 1
    max_node_count = 5
    preemptible    = false
  }
}

# Storage lifecycle rules
variable "storage_lifecycle_rules" {
  description = "Lifecycle rules for Cloud Storage buckets"
  type = list(object({
    action = object({
      type          = string
      storage_class = optional(string)
    })
    condition = object({
      age                   = optional(number)
      created_before        = optional(string)
      with_state           = optional(string)
      matches_storage_class = optional(list(string))
    })
  }))
  default = [
    {
      action = {
        type          = "SetStorageClass"
        storage_class = "NEARLINE"
      }
      condition = {
        age = 30
      }
    },
    {
      action = {
        type          = "SetStorageClass"
        storage_class = "COLDLINE"
      }
      condition = {
        age = 90
      }
    },
    {
      action = {
        type = "Delete"
      }
      condition = {
        age = 365
      }
    }
  ]
}

# Composer Airflow configuration
variable "composer_airflow_config" {
  description = "Airflow configuration for Cloud Composer"
  type        = map(string)
  default = {
    "webserver-worker_timeout"                = "180s"
    "webserver-worker_refresh_batch_size"     = "1"
    "webserver-worker_refresh_interval"       = "30s"
    "scheduler-catchup_by_default"            = "False"
    "scheduler-dag_dir_list_interval"         = "300"
    "scheduler-max_threads"                   = "2"
    "celery-worker_concurrency"              = "8"
    "core-dags_are_paused_at_creation"       = "True"
    "core-max_active_runs_per_dag"           = "2"
    "email-email_backend"                    = "airflow.utils.email.send_email_smtp"
  }
}

# Monitoring configuration
variable "monitoring_notification_emails" {
  description = "Email addresses for monitoring notifications"
  type        = list(string)
  default     = []
}

# SSL certificate domains
variable "ssl_certificate_domains" {
  description = "Domains for SSL certificates"
  type        = list(string)
  default     = []
}

# Cloud SQL configuration
variable "enable_cloud_sql" {
  description = "Enable Cloud SQL instance"
  type        = bool
  default     = false
}

variable "cloud_sql_database_version" {
  description = "Cloud SQL database version"
  type        = string
  default     = "POSTGRES_15"
}

variable "cloud_sql_tier" {
  description = "Cloud SQL machine type"
  type        = string
  default     = "db-custom-2-7680"
}

# Secret configurations
variable "secret_configs" {
  description = "Secret Manager secret configurations"
  type = map(object({
    secret_data = string
    labels      = optional(map(string), {})
  }))
  default = {}
  sensitive = true
}

# Backup configuration
variable "backup_schedule" {
  description = "Backup schedule configuration"
  type = object({
    frequency           = string  # daily, weekly, monthly
    retention_days      = number
    backup_window_hours = list(number)
  })
  default = {
    frequency           = "daily"
    retention_days      = 30
    backup_window_hours = [2, 4]  # 2 AM to 4 AM
  }
}

# Environment-specific configurations
locals {
  environment_configs = {
    dev = {
      dataproc_workers = {
        min = 2
        max = 4
      }
      gke_nodes = {
        min = 1
        max = 3
      }
      enable_deletion_protection = false
      enable_binary_logging      = false
      backup_retention_days      = 7
    }
    staging = {
      dataproc_workers = {
        min = 2
        max = 6
      }
      gke_nodes = {
        min = 2
        max = 4
      }
      enable_deletion_protection = true
      enable_binary_logging      = true
      backup_retention_days      = 14
    }
    prod = {
      dataproc_workers = {
        min = 3
        max = 10
      }
      gke_nodes = {
        min = 3
        max = 8
      }
      enable_deletion_protection = true
      enable_binary_logging      = true
      backup_retention_days      = 30
    }
  }
}

# Environment-specific resource sizing
variable "resource_sizing" {
  description = "Environment-specific resource sizing"
  type = map(object({
    bigquery_slots     = number
    composer_nodes     = number
    vertex_ai_nodes    = number
    storage_quota_gb   = number
  }))
  default = {
    dev = {
      bigquery_slots   = 100
      composer_nodes   = 3
      vertex_ai_nodes  = 2
      storage_quota_gb = 1000
    }
    staging = {
      bigquery_slots   = 500
      composer_nodes   = 5
      vertex_ai_nodes  = 3
      storage_quota_gb = 5000
    }
    prod = {
      bigquery_slots   = 2000
      composer_nodes   = 7
      vertex_ai_nodes  = 5
      storage_quota_gb = 20000
    }
  }
}

# Data retention policies
variable "data_retention_policies" {
  description = "Data retention policies by environment"
  type = map(object({
    raw_data_days       = number
    processed_data_days = number
    model_data_days     = number
    log_data_days       = number
  }))
  default = {
    dev = {
      raw_data_days       = 30
      processed_data_days = 60
      model_data_days     = 90
      log_data_days       = 7
    }
    staging = {
      raw_data_days       = 90
      processed_data_days = 180
      model_data_days     = 365
      log_data_days       = 30
    }
    prod = {
      raw_data_days       = 365
      processed_data_days = 2555  # 7 years for regulatory compliance
      model_data_days     = 2555  # 7 years for model audit trail
      log_data_days       = 90
    }
  }
}

# Network security settings
variable "network_security" {
  description = "Network security configuration"
  type = object({
    allowed_ip_ranges    = list(string)
    enable_private_nodes = bool
    enable_network_policy = bool
    authorized_networks  = list(object({
      cidr_block   = string
      display_name = string
    }))
  })
  default = {
    allowed_ip_ranges     = ["10.0.0.0/8"]
    enable_private_nodes  = true
    enable_network_policy = true
    authorized_networks   = []
  }
}

# Compliance and audit settings
variable "compliance_settings" {
  description = "Compliance and audit configuration"
  type = object({
    enable_audit_logs     = bool
    enable_data_lineage   = bool
    enable_access_transparency = bool
    retention_policy_days = number
  })
  default = {
    enable_audit_logs         = true
    enable_data_lineage       = true
    enable_access_transparency = true
    retention_policy_days     = 365
  }
}