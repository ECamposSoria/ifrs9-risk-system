output "bigquery_raw_dataset_id" {
  description = "Raw dataset ID (contains loan_portfolio table)"
  value       = google_bigquery_dataset.raw.dataset_id
}

output "bigquery_analytics_dataset_id" {
  description = "Analytics dataset ID (contains Looker-ready views)"
  value       = google_bigquery_dataset.analytics.dataset_id
}

output "bigquery_ml_dataset_id" {
  description = "ML dataset ID"
  value       = google_bigquery_dataset.ml.dataset_id
}

output "artifacts_bucket_name" {
  description = "GCS bucket name for job/model artifacts"
  value       = google_storage_bucket.artifacts.name
}

output "cloud_run_job_name" {
  description = "Cloud Run job name"
  value       = google_cloud_run_v2_job.portfolio_batch.name
}

output "cloud_run_job_region" {
  description = "Cloud Run job region"
  value       = google_cloud_run_v2_job.portfolio_batch.location
}

output "cloud_run_service_account_email" {
  description = "Service account email used by the Cloud Run job"
  value       = google_service_account.cloud_run_job.email
}

output "vertex_training_service_account_email" {
  description = "Service account email for Vertex training jobs (if enabled)"
  value       = var.enable_vertex_training_iam ? google_service_account.vertex_training[0].email : null
}

