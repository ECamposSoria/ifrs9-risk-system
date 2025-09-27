variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "region" {
  description = "Cloud SQL region"
  type        = string
}

variable "labels" {
  description = "Labels for Cloud SQL resources"
  type        = map(string)
  default     = {}
}

variable "network_name" {
  description = "VPC network for private service access"
  type        = string
}

variable "database_version" {
  description = "Cloud SQL database version"
  type        = string
}

variable "tier" {
  description = "Machine tier"
  type        = string
}
