locals {
  plan_name   = lower(replace("ifrs9-${var.environment}-cluster-backup", "_", "-"))
  today       = formatdate("2006-01-02", timestamp())
  today_parts = split("-", local.today)
  start_year  = tonumber(local.today_parts[0])
  start_month = tonumber(local.today_parts[1])
  start_day   = tonumber(local.today_parts[2])
  start_hour  = var.backup_schedule.backup_window_hours[0]
  cron_schedule = (
    lower(var.backup_schedule.frequency) == "daily" ?
    format("0 %d * * *", local.start_hour) :
    lower(var.backup_schedule.frequency) == "weekly" ?
    format("0 %d * * 1", local.start_hour) :
    format("0 %d 1 * *", local.start_hour)
  )
}

resource "google_gke_backup_backup_plan" "primary" {
  provider = google-beta

  name     = local.plan_name
  project  = var.project_id
  location = var.region
  cluster  = var.gke_cluster_id
  labels   = var.labels

  retention_policy {
    backup_retain_days      = var.backup_schedule.retention_days
    backup_delete_lock_days = var.backup_schedule.retention_days
  }

  backup_schedule {
    cron_schedule = local.cron_schedule
  }

  backup_config {
    all_namespaces      = true
    include_volume_data = true
    include_secrets     = true
  }
}

resource "google_storage_transfer_job" "gcs" {
  for_each = { for pair in var.gcs_backup_pairs : "${pair.source_bucket}->${pair.destination_bucket}" => pair }

  project     = var.project_id
  description = "IFRS9 ${var.environment} storage backup ${each.value.source_bucket}"
  status      = "ENABLED"

  schedule {
    schedule_start_date {
      year  = local.start_year
      month = local.start_month
      day   = local.start_day
    }
    start_time_of_day {
      hours   = local.start_hour
      minutes = 0
      seconds = 0
      nanos   = 0
    }
  }

  transfer_spec {
    gcs_data_source {
      bucket_name = each.value.source_bucket
    }

    gcs_data_sink {
      bucket_name = each.value.destination_bucket
    }

    dynamic "object_conditions" {
      for_each = length(coalesce(lookup(each.value, "include_prefixes", null), [])) > 0 ? [coalesce(lookup(each.value, "include_prefixes", null), [])] : []
      content {
        include_prefixes = coalesce(lookup(each.value, "include_prefixes", null), [])
      }
    }

    transfer_options {
      overwrite_objects_already_existing_in_sink = true
      delete_objects_from_source_after_transfer  = false
      delete_objects_unique_in_sink              = false
    }
  }
}
