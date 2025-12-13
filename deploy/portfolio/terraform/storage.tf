locals {
  artifacts_bucket_name = replace(lower("${var.project_id}-ifrs9-portfolio-artifacts-${var.environment}"), "_", "-")
}

resource "google_storage_bucket" "artifacts" {
  name     = local.artifacts_bucket_name
  project  = var.project_id
  location = var.region

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true

  lifecycle_rule {
    condition {
      age = var.artifact_bucket_retention_days
    }
    action {
      type = "Delete"
    }
  }

  labels = local.common_labels

  depends_on = [google_project_service.required_apis["storage"]]
}

