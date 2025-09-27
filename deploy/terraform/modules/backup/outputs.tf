output "backup_plan_name" {
  description = "Name of the GKE backup plan"
  value       = google_gke_backup_backup_plan.primary.name
}

output "backup_schedule" {
  description = "Cron schedule used for backups"
  value       = google_gke_backup_backup_plan.primary.backup_schedule[0].cron_schedule
}

output "storage_transfer_job_names" {
  description = "Storage Transfer Service job names"
  value       = [for job in google_storage_transfer_job.gcs : job.name]
}
