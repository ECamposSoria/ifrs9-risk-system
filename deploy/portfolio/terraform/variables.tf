variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Default region for resources (GCS, IAM, etc.)"
  type        = string
  default     = "southamerica-east1"
}

variable "environment" {
  description = "Environment suffix used in dataset names"
  type        = string
  default     = "staging"
}

variable "cloud_run_region" {
  description = "Region for the Cloud Run Job (often us-central1 for free-tier friendly demos)"
  type        = string
  default     = "us-central1"
}

variable "bigquery_location" {
  description = "Location for BigQuery datasets (must match the dataset location used by jobs)"
  type        = string
  default     = "southamerica-east1"
}

variable "cloud_run_job_image" {
  description = "Container image URI for the Cloud Run batch job (must be a pinned tag)"
  type        = string
}

variable "ifrs9_record_count" {
  description = "Synthetic portfolio size"
  type        = number
  default     = 5000
}

variable "ifrs9_seed" {
  description = "Random seed for synthetic data generation"
  type        = number
  default     = 42
}

variable "gcs_object_prefix" {
  description = "Prefix for objects uploaded to the artifacts bucket"
  type        = string
  default     = "ifrs9-portfolio"
}

variable "artifact_bucket_retention_days" {
  description = "Delete artifacts older than this many days"
  type        = number
  default     = 30
}

variable "enable_vertex_training_iam" {
  description = "Provision IAM bindings for a Vertex AI training-only service account"
  type        = bool
  default     = true
}

variable "labels" {
  description = "Labels applied to created resources"
  type        = map(string)
  default     = {}
}

