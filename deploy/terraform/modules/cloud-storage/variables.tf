variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
}

variable "region" {
  description = "Primary region for the buckets"
  type        = string
}

variable "labels" {
  description = "Labels applied to buckets"
  type        = map(string)
  default     = {}
}

variable "kms_key_id" {
  description = "KMS key used for default encryption"
  type        = string
}

variable "lifecycle_rules" {
  description = "Lifecycle rules applied to buckets"
  type = list(object({
    action = object({
      type          = string
      storage_class = optional(string)
    })
    condition = object({
      age                   = optional(number)
      created_before        = optional(string)
      with_state            = optional(string)
      matches_storage_class = optional(list(string))
    })
  }))
  default = []
}

variable "bucket_retention_days" {
  description = "Retention period in days enforced on managed buckets"
  type        = number
  default     = 365
}
