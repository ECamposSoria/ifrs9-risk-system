locals {
  base_name = replace(lower("ifrs9-${var.project_id}-${var.environment}"), "_", "-")
  buckets = {
    raw       = "${local.base_name}-raw"
    processed = "${local.base_name}-processed"
    models    = "${local.base_name}-models"
    artifacts = "${local.base_name}-artifacts"
    backups   = "${local.base_name}-backups"
  }
}

resource "google_storage_bucket" "managed" {
  for_each = local.buckets

  name                        = each.value
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  labels                      = var.labels
  public_access_prevention    = "enforced"
  force_destroy               = false

  encryption {
    default_kms_key_name = var.kms_key_id
  }

  versioning {
    enabled = true
  }

  retention_policy {
    retention_period = var.bucket_retention_days * 24 * 60 * 60
    is_locked        = false
  }

  dynamic "lifecycle_rule" {
    for_each = var.lifecycle_rules
    content {
      action {
        type          = lifecycle_rule.value.action.type
        storage_class = lookup(lifecycle_rule.value.action, "storage_class", null)
      }
      condition {
        age                   = lookup(lifecycle_rule.value.condition, "age", null)
        created_before        = lookup(lifecycle_rule.value.condition, "created_before", null)
        with_state            = lookup(lifecycle_rule.value.condition, "with_state", null)
        matches_storage_class = lookup(lifecycle_rule.value.condition, "matches_storage_class", null)
      }
    }
  }
}
