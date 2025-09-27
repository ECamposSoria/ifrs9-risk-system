locals {
  id_map = {
    raw       = google_bigquery_dataset.raw_dataset.dataset_id
    processed = google_bigquery_dataset.processed_dataset.dataset_id
    analytics = google_bigquery_dataset.analytics_dataset.dataset_id
    ml        = google_bigquery_dataset.ml_dataset.dataset_id
  }
}

output "raw_dataset_id" {
  description = "Dataset ID for raw data"
  value       = local.id_map.raw
}

output "processed_dataset_id" {
  description = "Dataset ID for processed data"
  value       = local.id_map.processed
}

output "analytics_dataset_id" {
  description = "Dataset ID for analytics"
  value       = local.id_map.analytics
}

output "ml_dataset_id" {
  description = "Dataset ID for ML features"
  value       = local.id_map.ml
}

output "dataset_ids" {
  description = "Map of dataset IDs"
  value       = local.id_map
}
