variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "region" {
  description = "Regional location for the GKE cluster"
  type        = string
}

variable "labels" {
  description = "Labels applied to the cluster"
  type        = map(string)
  default     = {}
}

variable "network_name" {
  description = "VPC network name"
  type        = string
}

variable "subnet_name" {
  description = "Subnet name"
  type        = string
}

variable "pod_secondary_range_name" {
  description = "Secondary range name allocated for GKE pods"
  type        = string
}

variable "service_secondary_range_name" {
  description = "Secondary range name allocated for GKE services"
  type        = string
}

variable "node_service_account_email" {
  description = "Service account used by the GKE node pool"
  type        = string
}

variable "node_config" {
  description = "Node pool configuration input"
  type = object({
    machine_type   = string
    disk_size_gb   = number
    disk_type      = string
    min_node_count = number
    max_node_count = number
    preemptible    = bool
    image_type     = optional(string)
  })
}

variable "node_labels" {
  description = "Additional labels applied to GKE nodes"
  type        = map(string)
  default     = {}
}

variable "node_tags" {
  description = "Network tags assigned to node VMs"
  type        = list(string)
  default     = []
}

variable "node_metadata" {
  description = "Custom metadata applied to node VMs"
  type        = map(string)
  default     = {}
}

variable "release_channel" {
  description = "GKE release channel"
  type        = string
  default     = "REGULAR"
}

variable "enable_private_endpoint" {
  description = "Expose only the private control plane endpoint"
  type        = bool
  default     = false
}

variable "master_ipv4_cidr" {
  description = "CIDR range allocated for the private control plane"
  type        = string
  default     = "10.90.48.0/28"
}

variable "authorized_networks" {
  description = "List of authorized networks allowed to reach the control plane"
  type = list(object({
    cidr_block   = string
    display_name = optional(string)
  }))
  default = []
}

variable "enable_network_policy" {
  description = "Enable Calico network policy on the cluster"
  type        = bool
  default     = true
}

variable "maintenance_window_start" {
  description = "Start time for the daily maintenance window (HH:MM)"
  type        = string
  default     = "03:00"
}

variable "enable_backup_agent" {
  description = "Enable the GKE Backup agent add-on"
  type        = bool
  default     = true
}

variable "disable_http_load_balancing" {
  description = "Disable the Cloud HTTP Load Balancing add-on"
  type        = bool
  default     = false
}

variable "upgrade_max_surge" {
  description = "Maximum number of nodes that can be created during upgrades"
  type        = number
  default     = 1
}

variable "upgrade_max_unavailable" {
  description = "Maximum number of nodes that can be unavailable during upgrades"
  type        = number
  default     = 0
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection on the cluster"
  type        = bool
  default     = true
}

variable "logging_components" {
  description = "GKE logging components to enable"
  type        = list(string)
  default = [
    "SYSTEM_COMPONENTS",
    "WORKLOADS"
  ]
}

variable "monitoring_components" {
  description = "GKE monitoring components to enable"
  type        = list(string)
  default = [
    "SYSTEM_COMPONENTS",
    "POD",
    "DEPLOYMENT"
  ]
}
