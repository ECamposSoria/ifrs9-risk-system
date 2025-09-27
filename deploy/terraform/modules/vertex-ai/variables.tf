variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "region" {
  description = "Primary Vertex AI region"
  type        = string
}

variable "zone" {
  description = "Zone for the Workbench instance"
  type        = string
  default     = "southamerica-east1-b"
}

variable "labels" {
  description = "Labels for bookkeeping"
  type        = map(string)
  default     = {}
}

variable "service_account_email" {
  description = "Service account used for Vertex AI pipelines"
  type        = string
}

variable "network_self_link" {
  description = "Self link of the VPC network"
  type        = string
}

variable "subnet_self_link" {
  description = "Self link of the subnetwork"
  type        = string
}

variable "desired_state" {
  description = "Desired state of the Workbench instance"
  type        = string
  default     = "RUNNING"
}

variable "machine_type" {
  description = "Machine type for the Workbench VM"
  type        = string
  default     = "e2-standard-2"
}

variable "boot_disk" {
  description = "Boot disk configuration"
  type = object({
    disk_type    = string
    disk_size_gb = number
    kms_key      = optional(string)
  })
  default = {
    disk_type    = "PD_SSD"
    disk_size_gb = 100
    kms_key      = null
  }
}

variable "vm_image" {
  description = "VM image family used for the Workbench instance"
  type = object({
    project = string
    family  = string
  })
  default = {
    project = "workbench-images"
    family  = "common-cpu-notebooks"
  }
}

variable "instance_owners" {
  description = "Users granted ownership of the Workbench instance"
  type        = list(string)
  default     = []
}

variable "disable_proxy_access" {
  description = "Disable Vertex Workbench proxy access"
  type        = bool
  default     = false
}
