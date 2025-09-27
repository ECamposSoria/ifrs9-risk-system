variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "region" {
  description = "Composer region"
  type        = string
}

variable "labels" {
  description = "Labels applied to Composer resources"
  type        = map(string)
  default     = {}
}

variable "network_name" {
  description = "Network used by Composer"
  type        = string
}

variable "subnet_name" {
  description = "Subnet used by Composer"
  type        = string
}

variable "service_account_email" {
  description = "Service account assigned to Composer"
  type        = string
}

variable "airflow_config" {
  description = "Airflow configuration overrides"
  type        = map(string)
  default     = {}
}

variable "env_variables" {
  description = "Environment variables injected into Airflow"
  type        = map(string)
  default     = {}
}

variable "pypi_packages" {
  description = "Additional PyPI packages installed in Composer environment"
  type        = map(string)
  default     = {}
}

variable "image_version" {
  description = "Composer image version"
  type        = string
  default     = "composer-2.6.5-airflow-2.6.3"
}

variable "environment_size" {
  description = "Composer environment size"
  type        = string
  default     = "ENVIRONMENT_SIZE_SMALL"
}

variable "enable_private_endpoint" {
  description = "Restrict access to the private Composer endpoint"
  type        = bool
  default     = true
}

variable "master_ipv4_cidr" {
  description = "CIDR block for the Composer GKE control plane"
  type        = string
  default     = "10.90.64.0/28"
}

variable "cloud_sql_ipv4_cidr" {
  description = "CIDR block for Composer's Cloud SQL instance"
  type        = string
  default     = "10.90.80.0/24"
}

variable "composer_network_ipv4_cidr" {
  description = "CIDR block for Composer internal network"
  type        = string
  default     = "10.90.66.0/24"
}

variable "connection_subnetwork" {
  description = "Optional PSC subnetwork for Composer connection"
  type        = string
  default     = null
}

variable "scheduler_resources" {
  description = "Resource profile for the Airflow scheduler"
  type = object({
    cpu        = number
    memory_gb  = number
    storage_gb = number
  })
  default = {
    cpu        = 1
    memory_gb  = 2
    storage_gb = 8
  }
}

variable "web_server_resources" {
  description = "Resource profile for the Airflow web server"
  type = object({
    cpu        = number
    memory_gb  = number
    storage_gb = number
  })
  default = {
    cpu        = 1
    memory_gb  = 2
    storage_gb = 8
  }
}

variable "worker_resources" {
  description = "Resource profile for Airflow workers"
  type = object({
    cpu        = number
    memory_gb  = number
    storage_gb = number
    min_count  = number
    max_count  = number
  })
  default = {
    cpu        = 1
    memory_gb  = 4
    storage_gb = 10
    min_count  = 2
    max_count  = 3
  }
}

variable "node_tags" {
  description = "Network tags applied to Composer GKE nodes"
  type        = list(string)
  default     = []
}

variable "resilience_mode" {
  description = "Composer resilience mode"
  type        = string
  default     = "STANDARD_RESILIENCE"
}
