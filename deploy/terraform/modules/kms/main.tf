locals {
  key_ring_name = replace("ifrs9-${var.environment}-keys", "_", "-")
}

resource "google_kms_key_ring" "core" {
  name     = local.key_ring_name
  project  = var.project_id
  location = var.region

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_kms_crypto_key" "storage" {
  name            = "storage-default"
  key_ring        = google_kms_key_ring.core.id
  rotation_period = "7776000s" # 90 days
  purpose         = "ENCRYPT_DECRYPT"

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_kms_crypto_key" "bigquery" {
  name            = "bigquery-default"
  key_ring        = google_kms_key_ring.core.id
  rotation_period = "7776000s"
  purpose         = "ENCRYPT_DECRYPT"

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_kms_crypto_key" "compute" {
  name            = "compute-default"
  key_ring        = google_kms_key_ring.core.id
  rotation_period = "7776000s"
  purpose         = "ENCRYPT_DECRYPT"

  lifecycle {
    prevent_destroy = true
  }
}
