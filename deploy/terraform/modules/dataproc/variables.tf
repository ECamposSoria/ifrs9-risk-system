variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "region" {
  description = "Dataproc region"
  type        = string
}

variable "labels" {
  description = "Labels applied to the Dataproc cluster"
  type        = map(string)
  default     = {}
}

variable "subnet_name" {
  description = "Subnetwork used for Dataproc nodes"
  type        = string
}

variable "service_account_email" {
  description = "Service account used by the cluster"
  type        = string
}

variable "enable_autoscaling" {
  description = "Enable an autoscaling policy for the cluster"
  type        = bool
  default     = true
}

variable "min_workers" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 2
}

variable "max_workers" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 6
}

variable "master_disk_size_gb" {
  description = "Boot disk size for the Dataproc master node"
  type        = number
  default     = 100
}

variable "worker_disk_size_gb" {
  description = "Boot disk size for each Dataproc worker node"
  type        = number
  default     = 100
}
