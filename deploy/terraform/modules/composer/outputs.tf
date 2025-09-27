output "environment_name" {
  description = "Composer environment name"
  value       = google_composer_environment.primary.name
}

output "airflow_uri" {
  description = "Airflow web UI URI"
  value       = google_composer_environment.primary.config[0].airflow_uri
}

output "gcs_bucket" {
  description = "Composer DAG bucket"
  value       = element(split("/", replace(google_composer_environment.primary.config[0].dag_gcs_prefix, "gs://", "")), 0)
}

output "dag_folder" {
  description = "Path to the DAG folder"
  value       = google_composer_environment.primary.config[0].dag_gcs_prefix
}

output "service_account" {
  description = "Service account used by the Composer environment"
  value       = google_composer_environment.primary.config[0].node_config[0].service_account
}
