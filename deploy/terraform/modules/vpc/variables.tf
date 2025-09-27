variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment identifier"
  type        = string
}

variable "region" {
  description = "Primary region for the VPC subnet"
  type        = string
}

variable "labels" {
  description = "Common labels applied to resources"
  type        = map(string)
  default     = {}
}

variable "subnet_cidr" {
  description = "CIDR range for the primary subnet"
  type        = string
  default     = "10.90.0.0/20"
}

variable "pod_ip_cidr" {
  description = "Secondary CIDR range allocated for GKE Pod IPs"
  type        = string
  default     = "10.90.32.0/21"
}

variable "service_ip_cidr" {
  description = "Secondary CIDR range allocated for GKE Service IPs"
  type        = string
  default     = "10.90.40.0/22"
}

variable "firewall_target_tags" {
  description = "Additional network tags permitted for egress allow rules"
  type        = list(string)
  default     = []
}

variable "enable_nat_logging" {
  description = "Enable Cloud NAT logging"
  type        = bool
  default     = true
}

variable "nat_log_filter" {
  description = "Cloud NAT log filter (ALL, ERRORS_ONLY, TRANSLATIONS_ONLY)"
  type        = string
  default     = "ALL"
}

variable "subnet_flow_sampling" {
  description = "Flow log sampling rate (0.0 - 1.0)"
  type        = number
  default     = 0.5
}

variable "subnet_flow_aggregation_interval" {
  description = "Aggregation interval for VPC flow logs"
  type        = string
  default     = "INTERVAL_5_MIN"
}

variable "enforce_autopilot_drain" {
  description = "Ensure no Autopilot maintenance VMs exist before updating secondary ranges"
  type        = bool
  default     = true
}

variable "autopilot_active_nodes" {
  description = "Number of Autopilot maintenance instances (gk3-*) observed"
  type        = number
  default     = 0
}

variable "autopilot_active_node_names" {
  description = "List of Autopilot maintenance instance names"
  type        = list(string)
  default     = []
}
