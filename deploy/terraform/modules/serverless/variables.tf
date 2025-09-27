variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment identifier"
  type        = string
}

variable "region" {
  description = "Region for serverless resources"
  type        = string
}

variable "labels" {
  description = "Labels applied to created resources"
  type        = map(string)
  default     = {}
}

variable "cloud_run_image" {
  description = "Container image executed by Cloud Run"
  type        = string
}

variable "cloud_run_timeout_seconds" {
  description = "Request timeout for the Cloud Run service"
  type        = number
  default     = 900
}

variable "schedule_cron" {
  description = "Cron schedule for Cloud Scheduler triggering the workflow"
  type        = string
}
