variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "region" {
  description = "Dataset location"
  type        = string
}

variable "labels" {
  description = "Labels applied to datasets"
  type        = map(string)
  default     = {}
}

variable "kms_key_id" {
  description = "KMS key used for CMEK encryption"
  type        = string
}

variable "dataset_owner_email" {
  description = "Dataset owner user email"
  type        = string
}

variable "service_accounts" {
  description = "Map of service account emails allowed to access datasets"
  type        = map(string)
  default     = {}
}

variable "pipeline_writers_group" {
  description = "Optional Google group granted write access for pipelines"
  type        = string
  default     = ""
}

variable "data_retention_policies" {
  description = "Retention policy configuration passed from the root module"
  type = map(object({
    raw_data_days       = number
    processed_data_days = number
    model_data_days     = number
    log_data_days       = number
  }))
}

variable "dashboard_viewers_group" {
  description = "Optional Google group granted read access to analytics datasets"
  type        = string
  default     = ""
}

variable "enable_tables" {
  description = "Create BigQuery tables, routines, and scheduled queries"
  type        = bool
  default     = true
}

variable "data_transfer_service_account" {
  description = "Service account email used by BigQuery Data Transfer scheduled queries"
  type        = string
  default     = ""
}
