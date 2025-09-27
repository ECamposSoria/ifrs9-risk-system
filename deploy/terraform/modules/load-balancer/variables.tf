variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "labels" {
  description = "Labels for load balancer resources"
  type        = map(string)
  default     = {}
}

variable "backend_services" {
  description = "Backend bucket configuration"
  type = map(object({
    bucket_name = string
    description = optional(string)
    enable_cdn  = optional(bool)
    paths       = optional(list(string))
  }))
}

variable "default_backend" {
  description = "Key of the backend that should receive default traffic"
  type        = string
  default     = null
}

variable "ssl_certificate_domains" {
  description = "Domains covered by the managed certificate"
  type        = list(string)
  default     = []
}
