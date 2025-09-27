locals {
  notification_map = { for email in var.notification_emails : email => replace(email, "@", "-") }
}

data "google_project" "current" {
  project_id = var.project_id
}

resource "google_monitoring_notification_channel" "email" {
  for_each = local.notification_map

  project      = var.project_id
  display_name = "ifrs9-${var.environment}-${each.value}"
  description  = "IFRS9 ${var.environment} budget and alert notifications"
  type         = "email"

  labels = {
    email_address = each.key
  }
}

resource "google_logging_project_sink" "audit_logs" {
  count = var.audit_dataset_id == "" ? 0 : 1

  name                   = "ifrs9-${var.environment}-audit-sink"
  project                = var.project_id
  destination            = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${var.audit_dataset_id}"
  filter                 = "logName=\"projects/${var.project_id}/logs/cloudaudit.googleapis.com%2Factivity\" OR logName=\"projects/${var.project_id}/logs/cloudaudit.googleapis.com%2Fdata_access\""
  unique_writer_identity = true
}

resource "google_bigquery_dataset_iam_member" "audit_sink_writer" {
  count = length(google_logging_project_sink.audit_logs) > 0 ? 1 : 0

  project    = var.project_id
  dataset_id = var.audit_dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.audit_logs[0].writer_identity
}

resource "google_billing_budget" "project_budget" {
  count = var.billing_account_id != "" && var.budget_monthly_amount > 0 ? 1 : 0

  billing_account = var.billing_account_id
  display_name    = upper("IFRS9 ${var.environment} budget")

  budget_filter {
    projects = [format("projects/%s", data.google_project.current.number)]
  }

  amount {
    specified_amount {
      currency_code = var.budget_currency_code
      units         = format("%d", floor(var.budget_monthly_amount))
    }
  }

  threshold_rules {
    threshold_percent = var.budget_alert_spend_threshold
  }

  threshold_rules {
    threshold_percent = var.budget_alert_spend_threshold
    spend_basis       = "FORECASTED_SPEND"
  }

  dynamic "all_updates_rule" {
    for_each = length(google_monitoring_notification_channel.email) > 0 ? [true] : []
    content {
      monitoring_notification_channels = [for channel in google_monitoring_notification_channel.email : channel.name]
      disable_default_iam_recipients   = true
    }
  }
}
