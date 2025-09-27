variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "region" {
  description = "Primary region for backup coordination"
  type        = string
}

variable "labels" {
  description = "Labels applied to backup resources"
  type        = map(string)
  default     = {}
}

variable "gke_cluster_id" {
  description = "Fully qualified ID of the GKE cluster to protect"
  type        = string
}

variable "gcs_backup_pairs" {
  description = "List of bucket pairs replicated via Storage Transfer Service"
  type = list(object({
    source_bucket      = string
    destination_bucket = string
    include_prefixes   = optional(list(string))
  }))
  default = []
}

variable "backup_schedule" {
  description = "Backup cadence configuration"
  type = object({
    frequency           = string
    retention_days      = number
    backup_window_hours = list(number)
  })
}
