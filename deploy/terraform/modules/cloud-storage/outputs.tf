locals {
  bucket_names = { for key, bucket in google_storage_bucket.managed : key => bucket.name }
}

output "raw_data_bucket_name" {
  description = "Name of the raw data bucket"
  value       = local.bucket_names.raw
}

output "processed_bucket_name" {
  description = "Name of the processed data bucket"
  value       = local.bucket_names.processed
}

output "models_bucket_name" {
  description = "Name of the ML models bucket"
  value       = local.bucket_names.models
}

output "artifacts_bucket_name" {
  description = "Name of the artifacts bucket"
  value       = local.bucket_names.artifacts
}

output "backups_bucket_name" {
  description = "Name of the backups bucket"
  value       = local.bucket_names.backups
}

output "bucket_names" {
  description = "Map of all bucket names"
  value       = local.bucket_names
}
