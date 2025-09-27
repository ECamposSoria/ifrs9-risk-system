output "notification_channel_ids" {
  description = "Email notification channel resource names"
  value       = [for channel in google_monitoring_notification_channel.email : channel.name]
}

output "budget_names" {
  description = "Billing budget resource names"
  value       = [for budget in google_billing_budget.project_budget : budget.name]
}

output "audit_log_sink" {
  description = "Audit log sink metadata"
  value = length(google_logging_project_sink.audit_logs) > 0 ? {
    name            = google_logging_project_sink.audit_logs[0].name
    destination     = google_logging_project_sink.audit_logs[0].destination
    writer_identity = google_logging_project_sink.audit_logs[0].writer_identity
  } : null
}
