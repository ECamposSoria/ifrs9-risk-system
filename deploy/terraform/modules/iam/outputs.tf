locals {
  emails = { for key, sa in google_service_account.service_accounts : key => sa.email }
}

output "dataproc_service_account_email" {
  description = "Service account email used by Dataproc"
  value       = local.emails.dataproc
}

output "composer_service_account_email" {
  description = "Service account email used by Cloud Composer"
  value       = local.emails.composer
}

output "vertex_service_account_email" {
  description = "Service account email used by Vertex AI"
  value       = local.emails.vertex
}

output "gke_service_account_email" {
  description = "Service account email used by the GKE node pool"
  value       = local.emails.gke
}

output "service_account_emails" {
  description = "Map of all created service account emails"
  value       = local.emails
}
