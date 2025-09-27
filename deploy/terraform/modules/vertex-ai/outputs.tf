output "workbench_instance_name" {
  description = "Vertex AI Workbench instance name"
  value       = google_workbench_instance.primary.name
}

output "workbench_location" {
  description = "Zone hosting the Workbench instance"
  value       = google_workbench_instance.primary.location
}

output "pipeline_service_account" {
  description = "Service account email used for Vertex pipelines"
  value       = var.service_account_email
}

output "model_registry_location" {
  description = "Location of the Vertex AI model registry"
  value       = var.region
}
